"""
Module 3 — Emergency Vehicle Routing
Bengaluru Road Network + Dynamic Risk-Aware Edge Weighting + A* Pathfinding

Pipeline:
  accident (lat, lng) + risk_score (from Module 2)
      ↓
  find nearest hospitals
      ↓
  re-weight road edges (avoid accident zone, avoid high-risk black spots, account for peak hour)
      ↓
  A* finds optimal route from each hospital
      ↓
  returns ETA + route coordinates + interactive map
"""

import os
import math
import pickle
import warnings
import networkx as nx
warnings.filterwarnings('ignore')

try:
    import osmnx as ox
    ox.settings.log_console = False
    ox.settings.use_cache   = True
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


# ── Bengaluru Black Spots (BBMP / Traffic Police reported high-accident junctions) ──

BENGALURU_BLACK_SPOTS = {
    "Silk Board Junction":              {"lat": 12.9175, "lng": 77.6231, "risk": 0.91},
    "Hebbal Flyover":                   {"lat": 13.0350, "lng": 77.5970, "risk": 0.85},
    "Marathahalli Bridge":              {"lat": 12.9591, "lng": 77.6971, "risk": 0.83},
    "KR Puram Junction":                {"lat": 13.0050, "lng": 77.6940, "risk": 0.82},
    "Tin Factory":                      {"lat": 12.9955, "lng": 77.6609, "risk": 0.80},
    "Madiwala":                         {"lat": 12.9236, "lng": 77.6175, "risk": 0.79},
    "Electronic City Flyover":          {"lat": 12.8458, "lng": 77.6600, "risk": 0.78},
    "Whitefield Main Road":             {"lat": 12.9698, "lng": 77.7500, "risk": 0.76},
    "Sarjapur Road Junction":           {"lat": 12.9102, "lng": 77.6835, "risk": 0.75},
    "Bannerghatta Road":                {"lat": 12.8933, "lng": 77.5975, "risk": 0.74},
    "Outer Ring Road Kadubeesanahalli": {"lat": 12.9360, "lng": 77.6970, "risk": 0.77},
    "Nagawara Junction":                {"lat": 13.0440, "lng": 77.6130, "risk": 0.73},
}

# ── Hospitals: fetched dynamically from OpenStreetMap (all real Bengaluru hospitals) ──
# Falls back to this curated list only if OSMnx/network is unavailable.

FALLBACK_HOSPITALS = {
    "Manipal Hospital (Old Airport Rd)": {"lat": 12.9592, "lng": 77.6474, "ambulances": 4},
    "Apollo BGS Hospital":               {"lat": 12.9250, "lng": 77.5938, "ambulances": 3},
    "Fortis Hospital Bannerghatta":      {"lat": 12.8907, "lng": 77.5973, "ambulances": 3},
    "Victoria Hospital":                 {"lat": 12.9663, "lng": 77.5757, "ambulances": 5},
    "MS Ramaiah Hospital":               {"lat": 13.0213, "lng": 77.5595, "ambulances": 3},
    "St. John's Medical College":        {"lat": 12.9400, "lng": 77.6210, "ambulances": 4},
    "Sakra World Hospital":              {"lat": 12.9574, "lng": 77.6958, "ambulances": 2},
    "Aster CMI Hospital":                {"lat": 13.0514, "lng": 77.5913, "ambulances": 3},
    "Narayana Health City":              {"lat": 12.8624, "lng": 77.6430, "ambulances": 4},
    "NIMHANS":                           {"lat": 12.9407, "lng": 77.5956, "ambulances": 2},
}

HOSPITALS_CACHE = os.path.join(os.path.dirname(__file__), "bengaluru_hospitals.pkl")
HOSPITALS = None  # populated by load_hospitals() at startup


def load_hospitals(force_rebuild: bool = False) -> dict:
    """
    Load all real Bengaluru hospitals from OpenStreetMap via OSMnx.
    Cached to disk after first fetch. Falls back to a small curated
    list if OSMnx/network is unavailable.
    """
    global HOSPITALS

    if os.path.exists(HOSPITALS_CACHE) and not force_rebuild:
        with open(HOSPITALS_CACHE, "rb") as f:
            HOSPITALS = pickle.load(f)
        print(f"Loaded {len(HOSPITALS)} hospitals from cache")
        return HOSPITALS

    if not OSMNX_AVAILABLE:
        print("OSMnx not available — using fallback hospital list")
        HOSPITALS = FALLBACK_HOSPITALS
        return HOSPITALS

    try:
        print("Fetching all hospitals in Bengaluru from OpenStreetMap...")
        tags = {"amenity": "hospital"}
        gdf = ox.features_from_place("Bengaluru, Karnataka, India", tags)

        hospitals = {}
        for idx, row in gdf.iterrows():
            name = row.get("name")
            if not name or (isinstance(name, float)):  # skip unnamed / NaN
                continue
            geom = row.geometry
            centroid = geom.centroid if geom.geom_type != "Point" else geom
            lat, lng = centroid.y, centroid.x

            # Skip duplicates (same hospital tagged as both way and node)
            if name in hospitals:
                continue

            hospitals[name] = {
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "ambulances": 2,  # OSM doesn't carry ambulance counts; assume baseline
            }

        if len(hospitals) < 5:
            print(f"Only found {len(hospitals)} hospitals — merging with fallback list")
            hospitals = {**FALLBACK_HOSPITALS, **hospitals}

        with open(HOSPITALS_CACHE, "wb") as f:
            pickle.dump(hospitals, f)

        print(f"Fetched and cached {len(hospitals)} real Bengaluru hospitals")
        HOSPITALS = hospitals
        return HOSPITALS

    except Exception as e:
        print(f"Hospital fetch failed ({e}) — using fallback list")
        HOSPITALS = FALLBACK_HOSPITALS
        return HOSPITALS

GRAPH_CACHE = os.path.join(os.path.dirname(__file__), "bengaluru_graph.pkl")


# ── Haversine distance ─────────────────────────────────────────────────────────

def haversine(lat1, lng1, lat2, lng2) -> float:
    """Great-circle distance in km between two GPS coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_location_risk(lat: float, lng: float, radius_km: float = 0.5) -> float:
    """Returns historical black-spot risk if near a known danger zone, else 0.50 baseline."""
    for name, spot in BENGALURU_BLACK_SPOTS.items():
        if haversine(lat, lng, spot["lat"], spot["lng"]) <= radius_km:
            return spot["risk"]
    return 0.50


def find_nearest_hospitals(lat: float, lng: float, top_k: int = 3) -> list:
    global HOSPITALS
    if HOSPITALS is None:
        load_hospitals()

    results = []
    for name, info in HOSPITALS.items():
        dist = haversine(lat, lng, info["lat"], info["lng"])
        results.append({
            "name": name, "lat": info["lat"], "lng": info["lng"],
            "ambulances": info["ambulances"], "distance_km": round(dist, 2),
        })
    results.sort(key=lambda x: x["distance_km"])
    return results[:top_k]


# ── Road graph ────────────────────────────────────────────────────────────────

def load_or_build_graph(force_rebuild: bool = False):
    """Load Bengaluru road graph from cache, download via OSMnx, or fall back to a simplified graph."""
    if os.path.exists(GRAPH_CACHE) and not force_rebuild:
        print("Loading Bengaluru road graph from cache...")
        with open(GRAPH_CACHE, "rb") as f:
            G = pickle.load(f)
        print(f"Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
        return G

    if OSMNX_AVAILABLE:
        try:
            print("Downloading Bengaluru road graph from OpenStreetMap...")
            print("(Takes ~2-3 minutes on first run, cached after that)")
            G = ox.graph_from_place("Bengaluru, Karnataka, India", network_type="drive", simplify=True)
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            with open(GRAPH_CACHE, "wb") as f:
                pickle.dump(G, f)
            print(f"Graph saved: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
            return G
        except Exception as e:
            print(f"OSMnx download failed ({e}) — using simplified graph")
            return build_simplified_graph()
    else:
        print("OSMnx not installed — using simplified graph")
        return build_simplified_graph()


def build_simplified_graph():
    """Simplified Bengaluru graph using real major-junction coordinates. No internet required."""
    G = nx.DiGraph()

    nodes = {
        "silk_board": (12.9175, 77.6231), "hsr_layout": (12.9116, 77.6389),
        "koramangala": (12.9352, 77.6245), "btm_layout": (12.9165, 77.6101),
        "madiwala": (12.9236, 77.6175), "electronic_city": (12.8458, 77.6600),
        "marathahalli": (12.9591, 77.6971), "whitefield": (12.9698, 77.7500),
        "kr_puram": (13.0050, 77.6940), "tin_factory": (12.9955, 77.6609),
        "hebbal": (13.0350, 77.5970), "yeshwanthpur": (13.0214, 77.5512),
        "majestic": (12.9767, 77.5713), "mg_road": (12.9752, 77.6085),
        "indiranagar": (12.9784, 77.6408), "nagawara": (13.0440, 77.6130),
        "airport_road": (12.9592, 77.6474), "st_johns": (12.9400, 77.6210),
        "victoria_hosp": (12.9663, 77.5757), "ms_ramaiah": (13.0213, 77.5595),
        "fortis_banner": (12.8907, 77.5973), "manipal_hosp": (12.9592, 77.6474),
        "apollo_bgs": (12.9250, 77.5938), "sakra_hosp": (12.9574, 77.6958),
        "aster_cmi": (13.0514, 77.5913),
    }
    for node_id, (lat, lng) in nodes.items():
        G.add_node(node_id, y=lat, x=lng)

    edges = [
        ("silk_board","koramangala",2.5), ("silk_board","btm_layout",2.0),
        ("silk_board","madiwala",1.2), ("silk_board","hsr_layout",2.8),
        ("hsr_layout","marathahalli",5.0), ("koramangala","mg_road",3.5),
        ("koramangala","indiranagar",3.0), ("koramangala","st_johns",2.5),
        ("btm_layout","madiwala",1.5), ("btm_layout","apollo_bgs",4.0),
        ("madiwala","electronic_city",7.0), ("madiwala","fortis_banner",3.5),
        ("electronic_city","fortis_banner",4.5), ("marathahalli","whitefield",6.5),
        ("marathahalli","kr_puram",5.5), ("marathahalli","indiranagar",4.0),
        ("marathahalli","sakra_hosp",2.0), ("whitefield","kr_puram",8.0),
        ("kr_puram","tin_factory",3.0), ("tin_factory","hebbal",5.5),
        ("tin_factory","nagawara",4.5), ("hebbal","yeshwanthpur",8.0),
        ("hebbal","aster_cmi",2.5), ("hebbal","ms_ramaiah",3.0),
        ("yeshwanthpur","majestic",4.0), ("yeshwanthpur","ms_ramaiah",2.5),
        ("majestic","mg_road",2.5), ("majestic","victoria_hosp",1.5),
        ("mg_road","indiranagar",3.0), ("mg_road","airport_road",4.5),
        ("indiranagar","airport_road",2.0), ("indiranagar","nagawara",5.5),
        ("airport_road","manipal_hosp",0.5), ("nagawara","aster_cmi",3.5),
        ("nagawara","hebbal",5.0), ("st_johns","apollo_bgs",3.5),
        ("st_johns","victoria_hosp",4.0), ("ms_ramaiah","aster_cmi",4.5),
    ]
    for u, v, dist_km in edges:
        travel_time = (dist_km / 40.0) * 3600  # 40 km/h avg urban speed
        G.add_edge(u, v, length=dist_km*1000, travel_time=travel_time, speed_kph=40)
        G.add_edge(v, u, length=dist_km*1000, travel_time=travel_time, speed_kph=40)

    print(f"Simplified graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── Dynamic risk-aware edge re-weighting ──────────────────────────────────────

def apply_dynamic_weights(G, accident_lat, accident_lng, risk_score, hour):
    """
    Re-weight edges based on:
    - Proximity to accident (avoid the immediate zone)
    - Peak hour congestion
    - Proximity to known black spots (only matters when risk is high)
    - Night-time speed advantage

    Works with both DiGraph (simplified fallback) and MultiDiGraph (real OSMnx graph).
    """
    G = G.copy()
    is_peak  = (8 <= hour <= 10) or (17 <= hour <= 20)
    is_night = (hour >= 22) or (hour <= 5)
    is_multigraph = G.is_multigraph()

    for u, v, k, data in G.edges(keys=True, data=True) if is_multigraph else \
                          ((u, v, None, data) for u, v, data in G.edges(data=True)):
        base_time = data.get("travel_time", 300)
        u_data, v_data = G.nodes[u], G.nodes[v]
        mid_lat = (u_data.get("y", accident_lat) + v_data.get("y", accident_lat)) / 2
        mid_lng = (u_data.get("x", accident_lng) + v_data.get("x", accident_lng)) / 2

        dist_from_accident = haversine(accident_lat, accident_lng, mid_lat, mid_lng)
        factor = 1.0

        if dist_from_accident < 0.5:
            factor *= 3.0
        elif dist_from_accident < 1.5:
            factor *= 1.8

        if is_peak:
            factor *= 1.4
        if is_night:
            factor *= 0.85

        if risk_score > 0.8:
            spot_risk = get_location_risk(mid_lat, mid_lng, radius_km=1.0)
            if spot_risk > 0.75:
                factor *= 1.6

        if is_multigraph:
            G[u][v][k]["weight"] = base_time * factor
        else:
            G[u][v]["weight"] = base_time * factor

    return G


def nearest_node(G, lat, lng):
    best_node, best_dist = None, float("inf")
    for node, data in G.nodes(data=True):
        dist = haversine(lat, lng, data.get("y", 0), data.get("x", 0))
        if dist < best_dist:
            best_dist, best_node = dist, node
    return best_node


def _get_edge_data(G, u, v):
    """Returns a single edge-data dict for u->v, handling both graph types.
    For MultiDiGraph, picks the edge with the lowest weight (or travel_time)."""
    if G.is_multigraph():
        edges = G[u][v]  # dict of {key: data}
        return min(edges.values(), key=lambda d: d.get("weight", d.get("travel_time", 1e9)))
    return G[u][v]


def astar_route(G, origin_lat, origin_lng, dest_lat, dest_lng) -> dict:
    origin_node = nearest_node(G, origin_lat, origin_lng)
    dest_node   = nearest_node(G, dest_lat, dest_lng)

    def heuristic(u, v):
        return haversine(G.nodes[u].get("y",0), G.nodes[u].get("x",0),
                          G.nodes[v].get("y",0), G.nodes[v].get("x",0)) / 40 * 3600

    try:
        path = nx.astar_path(G, origin_node, dest_node, heuristic=heuristic, weight="weight")
        total_time, total_dist, coords = 0, 0, []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge = _get_edge_data(G, u, v)
            total_time += edge.get("weight", edge.get("travel_time", 60))
            total_dist += edge.get("length", 500)
            coords.append([G.nodes[u].get("y",0), G.nodes[u].get("x",0)])
        coords.append([G.nodes[path[-1]].get("y",0), G.nodes[path[-1]].get("x",0)])

        return {
            "path_nodes": path, "coordinates": coords,
            "distance_km": round(total_dist/1000, 2),
            "eta_minutes": round(total_time/60, 1),
        }
    except nx.NetworkXNoPath:
        dist = haversine(origin_lat, origin_lng, dest_lat, dest_lng)
        eta = (dist / 40) * 60
        return {
            "path_nodes": [origin_node, dest_node],
            "coordinates": [[origin_lat, origin_lng], [dest_lat, dest_lng]],
            "distance_km": round(dist, 2), "eta_minutes": round(eta, 1),
        }


# ── Main routing function ─────────────────────────────────────────────────────

def route_emergency(accident_lat, accident_lng, risk_score, hour, G=None) -> dict:
    """Full pipeline: black-spot check -> nearest hospitals -> dynamic weights -> A*."""
    if G is None:
        G = load_or_build_graph()

    print(f"\n🚨 Accident at ({accident_lat:.4f}, {accident_lng:.4f}) | Risk: {risk_score:.2f} | Hour: {hour:02d}:00")

    black_spot_risk = get_location_risk(accident_lat, accident_lng)
    if black_spot_risk > 0.5:
        print(f"   📍 Near a known black spot (risk={black_spot_risk:.2f})")

    hospitals = find_nearest_hospitals(accident_lat, accident_lng, top_k=3)
    G_weighted = apply_dynamic_weights(G, accident_lat, accident_lng, risk_score, hour)

    routes = []
    for h in hospitals:
        r = astar_route(G_weighted, h["lat"], h["lng"], accident_lat, accident_lng)
        routes.append({
            "hospital": h["name"], "hospital_lat": h["lat"], "hospital_lng": h["lng"],
            "ambulances": h["ambulances"], "distance_km": r["distance_km"],
            "eta_minutes": r["eta_minutes"], "coordinates": r["coordinates"],
        })
    routes.sort(key=lambda r: r["eta_minutes"])
    primary, alternate = routes[0], routes[1:]

    dispatch_msg = (f"DISPATCH: {primary['hospital']} | "
                     f"ETA {primary['eta_minutes']} min | {primary['distance_km']} km")
    print(f"✅ {dispatch_msg}")

    return {
        "nearest_hospitals": hospitals,
        "primary_route": primary,
        "alternate_routes": alternate,
        "black_spot_risk": black_spot_risk,
        "dispatch_message": dispatch_msg,
        "accident_location": {"lat": accident_lat, "lng": accident_lng},
    }


# ── Map generator ─────────────────────────────────────────────────────────────

def generate_map(routing_result: dict, output_path: str = None):
    try:
        import folium
    except ImportError:
        print("folium not installed. Run: pip install folium")
        return None

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "emergency_map.html")

    accident = routing_result["accident_location"]
    primary  = routing_result["primary_route"]

    m = folium.Map(location=[accident["lat"], accident["lng"]], zoom_start=13, tiles="OpenStreetMap")

    folium.Marker(
        [accident["lat"], accident["lng"]],
        popup=f"ACCIDENT<br>Risk: {routing_result['black_spot_risk']:.2f}",
        icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
    ).add_to(m)

    folium.Marker(
        [primary["hospital_lat"], primary["hospital_lng"]],
        popup=f"{primary['hospital']}<br>ETA: {primary['eta_minutes']} min",
        icon=folium.Icon(color="green", icon="plus-sign", prefix="glyphicon"),
    ).add_to(m)

    if len(primary["coordinates"]) >= 2:
        folium.PolyLine(
            primary["coordinates"], color="blue", weight=5, opacity=0.8,
            popup=f"{primary['eta_minutes']} min | {primary['distance_km']} km",
        ).add_to(m)

    for name, spot in BENGALURU_BLACK_SPOTS.items():
        folium.CircleMarker(
            [spot["lat"], spot["lng"]], radius=7,
            color="orange", fill=True, fill_color="orange", fill_opacity=0.5,
            popup=f"Black Spot: {name} ({spot['risk']:.2f})",
        ).add_to(m)

    for alt in routing_result["alternate_routes"]:
        folium.Marker(
            [alt["hospital_lat"], alt["hospital_lng"]],
            popup=f"{alt['hospital']}<br>ETA: {alt['eta_minutes']} min",
            icon=folium.Icon(color="lightgreen", icon="plus-sign", prefix="glyphicon"),
        ).add_to(m)

    m.save(output_path)
    print(f"🗺️  Map saved: {output_path}")
    return output_path


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Module 3 — Bengaluru Emergency Vehicle Routing")
    print("=" * 55)

    G = load_or_build_graph()
    load_hospitals()

    # Test 1: accident at Silk Board Junction, evening peak, high risk
    result = route_emergency(accident_lat=12.9175, accident_lng=77.6231,
                              risk_score=0.87, hour=18, G=G)
    print(f"\nPrimary: {result['primary_route']['hospital']}")
    print(f"ETA:     {result['primary_route']['eta_minutes']} min")
    print(f"Dist:    {result['primary_route']['distance_km']} km")

    generate_map(result)

    # Test 2: accident near Marathahalli, morning peak
    print("\n" + "="*55)
    result2 = route_emergency(accident_lat=12.9591, accident_lng=77.6971,
                               risk_score=0.72, hour=9, G=G)
    print(f"Primary: {result2['primary_route']['hospital']} | ETA: {result2['primary_route']['eta_minutes']} min")