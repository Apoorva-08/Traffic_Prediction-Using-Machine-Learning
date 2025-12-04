import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import requests
import datetime
import json
import time
from typing import Dict, List, Optional
import hashlib

# Set page configuration
st.set_page_config(
    page_title="Smart Traffic Prediction System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: linear-gradient(90deg, #d4edda, #ffffff);
        border-left: 4px solid #28a745;
    }
    
    .status-warning {
        background: linear-gradient(90deg, #fff3cd, #ffffff);
        border-left: 4px solid #ffc107;
    }
    
    .status-error {
        background: linear-gradient(90deg, #f8d7da, #ffffff);
        border-left: 4px solid #dc3545;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration Class - Only TomTom
class APIConfig:
    def __init__(self):
        # Your TomTom API key
        self.tomtom_api_key = "Yl8DGpQ4j318QFEd0G8AlYJgSyqNzW2o"

# Real-time Traffic Data Manager with TomTom
class TomTomTrafficManager:
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def _get_cache_key(self, *args):
        return hashlib.md5(str(args).encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key):
        if cache_key not in self.cache:
            return False
        return time.time() - self.cache[cache_key]['timestamp'] < self.cache_duration
    
    def get_traffic_flow_data(self, lat: float, lon: float) -> Dict:
        """Get real-time traffic flow data from TomTom API"""
        cache_key = self._get_cache_key("tomtom_flow", lat, lon)
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # TomTom Traffic Flow API
            url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            params = {
                'point': f"{lat},{lon}",
                'key': self.api_config.tomtom_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'flowSegmentData' in data:
                flow_data = data['flowSegmentData']
                
                current_speed = flow_data.get('currentSpeed', 0)
                free_flow_speed = flow_data.get('freeFlowSpeed', 50)
                
                # Calculate congestion level
                if free_flow_speed > 0:
                    speed_ratio = current_speed / free_flow_speed
                    congestion_level = max(0, min(100, (1 - speed_ratio) * 100))
                else:
                    congestion_level = 0
                
                result = {
                    'current_speed': current_speed,
                    'free_flow_speed': free_flow_speed,
                    'current_travel_time': flow_data.get('currentTravelTime', 0),
                    'free_flow_travel_time': flow_data.get('freeFlowTravelTime', 0),
                    'confidence': flow_data.get('confidence', 0.7),
                    'road_closure': flow_data.get('roadClosure', False),
                    'congestion_level': congestion_level,
                    'coordinates': flow_data.get('coordinates', {}),
                    'status': 'success'
                }
                
                # Cache the result
                self.cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time()
                }
                return result
            
            return {"error": "No traffic flow data available", "status": "no_data"}
            
        except requests.exceptions.RequestException as e:
            return {"error": f"TomTom Flow API error: {str(e)}", "status": "error"}
    
    def get_traffic_incidents(self, lat: float, lon: float, radius: int = 10000) -> Dict:
        """Get traffic incidents from TomTom API"""
        cache_key = self._get_cache_key("tomtom_incidents", lat, lon, radius)
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # TomTom Traffic Incidents API
            url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
            params = {
                'bbox': f"{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}",
                'fields': '{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,events{description,code},startTime,endTime}}}',
                'language': 'en-GB',
                'key': self.api_config.tomtom_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            incidents = []
            if 'incidents' in data:
                for incident in data['incidents']:
                    properties = incident.get('properties', {})
                    events = properties.get('events', [])
                    geometry = incident.get('geometry', {})
                    coordinates = geometry.get('coordinates', [])
                    
                    # Fix coordinate order
                    incident_lat = lat
                    incident_lon = lon
                    
                    if coordinates and len(coordinates) >= 2:
                        if isinstance(coordinates[0], list):
                            incident_lon, incident_lat = coordinates[0][0], coordinates[0][1]
                        else:
                            incident_lon, incident_lat = coordinates[0], coordinates[1]
                    
                    incident_info = {
                        'type': properties.get('iconCategory', 'Unknown'),
                        'magnitude': properties.get('magnitudeOfDelay', 0),
                        'description': events[0].get('description', 'Traffic incident') if events else 'Traffic incident',
                        'start_time': properties.get('startTime', ''),
                        'end_time': properties.get('endTime', ''),
                        'lat': incident_lat,
                        'lon': incident_lon
                    }
                    incidents.append(incident_info)
            
            result = {
                'incidents': incidents,
                'incident_count': len(incidents),
                'status': 'success'
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            return result
            
        except requests.exceptions.RequestException as e:
            return {"error": f"TomTom Incidents API error: {str(e)}", "status": "error"}
    
    def get_route_data(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Dict:
        """Get route data with traffic from TomTom API"""
        cache_key = self._get_cache_key("tomtom_route", start_lat, start_lon, end_lat, end_lon)
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # TomTom Routing API
            url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_lat},{start_lon}:{end_lat},{end_lon}/json"
            params = {
                'traffic': 'true',
                'routeType': 'fastest',
                'travelMode': 'car',
                'key': self.api_config.tomtom_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'routes' in data and data['routes']:
                route = data['routes'][0]
                summary = route.get('summary', {})
                
                # Also get route without traffic for comparison
                params_no_traffic = params.copy()
                params_no_traffic['traffic'] = 'false'
                
                response_no_traffic = requests.get(url, params=params_no_traffic, timeout=10)
                no_traffic_data = response_no_traffic.json()
                
                no_traffic_time = 0
                if 'routes' in no_traffic_data and no_traffic_data['routes']:
                    no_traffic_time = no_traffic_data['routes'][0]['summary'].get('travelTimeInSeconds', 0)
                
                travel_time_with_traffic = summary.get('travelTimeInSeconds', 0)
                delay_seconds = travel_time_with_traffic - no_traffic_time
                
                result = {
                    'distance_meters': summary.get('lengthInMeters', 0),
                    'distance_km': round(summary.get('lengthInMeters', 0) / 1000, 2),
                    'travel_time_seconds': travel_time_with_traffic,
                    'travel_time_minutes': round(travel_time_with_traffic / 60, 1),
                    'no_traffic_time_seconds': no_traffic_time,
                    'no_traffic_time_minutes': round(no_traffic_time / 60, 1),
                    'delay_seconds': delay_seconds,
                    'delay_minutes': round(delay_seconds / 60, 1),
                    'traffic_delay_percentage': round((delay_seconds / max(no_traffic_time, 1)) * 100, 1),
                    'status': 'success'
                }
                
                # Cache the result
                self.cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time()
                }
                return result
            
            return {"error": "No route found", "status": "no_route"}
            
        except requests.exceptions.RequestException as e:
            return {"error": f"TomTom Routing API error: {str(e)}", "status": "error"}

# Weather Data Manager - Only Simulation
class WeatherManager:
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        self.cache = {}
        self.cache_duration = 1800  # 30 minutes cache
    
    def _get_cache_key(self, city):
        return hashlib.md5(f"weather_{city}".encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key):
        if cache_key not in self.cache:
            return False
        return time.time() - self.cache[cache_key]['timestamp'] < self.cache_duration
    
    def get_current_weather(self, city: str) -> Dict:
        """Get simulated weather data"""
        cache_key = self._get_cache_key(city)
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        return self.simulate_weather(city)
    
    def simulate_weather(self, city: str) -> Dict:
        """Simulate weather data based on city and season"""
        seed_value = hash(f"{city}_{datetime.datetime.now().date()}") % 1000
        np.random.seed(seed_value)
        
        city_climates = {
            'New York': {'temp_base': 15, 'rain_prob': 0.3},
            'Los Angeles': {'temp_base': 22, 'rain_prob': 0.1},
            'Chicago': {'temp_base': 10, 'rain_prob': 0.4},
            'Houston': {'temp_base': 25, 'rain_prob': 0.4},
            'Phoenix': {'temp_base': 30, 'rain_prob': 0.05}
        }
        
        climate = city_climates.get(city, {'temp_base': 20, 'rain_prob': 0.2})
        
        month = datetime.datetime.now().month
        if month in [12, 1, 2]:  # Winter
            temp_adj = -10
        elif month in [6, 7, 8]:  # Summer
            temp_adj = 10
        else:  # Spring/Fall
            temp_adj = 0
        
        temperature = climate['temp_base'] + temp_adj + np.random.normal(0, 5)
        
        result = {
            'temperature': round(temperature, 1),
            'humidity': np.random.randint(40, 80),
            'pressure': np.random.randint(1000, 1020),
            'weather': np.random.choice(['Clear', 'Clouds', 'Rain', 'Snow'], p=[0.4, 0.3, 0.2, 0.1]),
            'description': 'Simulated weather data',
            'wind_speed': np.random.uniform(0, 10),
            'visibility': np.random.uniform(5, 15),
            'rain': np.random.exponential(2) if np.random.random() < climate['rain_prob'] else 0,
            'snow': np.random.exponential(1) if temperature < 0 and np.random.random() < 0.1 else 0,
            'status': 'simulated'
        }
        
        # Cache the result
        self.cache[self._get_cache_key(city)] = {
            'data': result,
            'timestamp': time.time()
        }
        return result

# Enhanced Traffic Predictor
class EnhancedTrafficPredictor:
    def __init__(self, traffic_manager: TomTomTrafficManager, weather_manager: WeatherManager):
        self.traffic_manager = traffic_manager
        self.weather_manager = weather_manager
    
    def get_enhanced_prediction(self, city: str, base_prediction: float) -> Dict:
        """Get enhanced prediction using TomTom real-time data and simulated weather"""
        city_coords = {
            'New York': [40.7128, -74.0060],
            'Los Angeles': [34.0522, -118.2437],
            'Chicago': [41.8781, -87.6298],
            'Houston': [29.7604, -95.3698],
            'Phoenix': [33.4484, -112.0740]
        }
        
        if city not in city_coords:
            return {"prediction": base_prediction, "factors": [], "status": "no_data"}
        
        lat, lon = city_coords[city]
        
        # Get real-time data
        weather_data = self.weather_manager.get_current_weather(city)
        flow_data = self.traffic_manager.get_traffic_flow_data(lat, lon)
        incidents_data = self.traffic_manager.get_traffic_incidents(lat, lon)
        
        # Adjust prediction
        adjusted_prediction = base_prediction
        adjustment_factors = []
        
        # Weather adjustments
        if weather_data.get('rain', 0) > 0:
            rain_factor = min(weather_data['rain'] * 5, 20)
            adjusted_prediction += rain_factor
            adjustment_factors.append(f"Rain: +{rain_factor:.1f}%")
        
        if weather_data.get('snow', 0) > 0:
            snow_factor = min(weather_data['snow'] * 8, 30)
            adjusted_prediction += snow_factor
            adjustment_factors.append(f"Snow: +{snow_factor:.1f}%")
        
        if weather_data.get('visibility', 10) < 5:
            visibility_factor = (5 - weather_data['visibility']) * 3
            adjusted_prediction += visibility_factor
            adjustment_factors.append(f"Low visibility: +{visibility_factor:.1f}%")
        
        # TomTom flow data adjustments
        if flow_data.get('status') == 'success':
            congestion_level = flow_data.get('congestion_level', 0)
            if congestion_level > 20:
                flow_adjustment = min(congestion_level * 0.8, 25)
                adjusted_prediction += flow_adjustment
                adjustment_factors.append(f"Real traffic flow: +{flow_adjustment:.1f}%")
            
            if flow_data.get('road_closure', False):
                adjusted_prediction += 15
                adjustment_factors.append("Road closure: +15.0%")
        
        # Incident adjustments
        if incidents_data.get('status') == 'success':
            incident_count = incidents_data.get('incident_count', 0)
            if incident_count > 0:
                incident_factor = min(incident_count * 12, 30)
                adjusted_prediction += incident_factor
                adjustment_factors.append(f"Traffic incidents ({incident_count}): +{incident_factor:.1f}%")
        
        # Cap at 100%
        adjusted_prediction = min(adjusted_prediction, 100)
        
        return {
            'prediction': adjusted_prediction,
            'base_prediction': base_prediction,
            'adjustment': adjusted_prediction - base_prediction,
            'factors': adjustment_factors,
            'weather_data': weather_data,
            'flow_data': flow_data,
            'incidents_data': incidents_data,
            'status': 'success'
        }

# Initialize managers
api_config = APIConfig()
traffic_manager = TomTomTrafficManager(api_config)
weather_manager = WeatherManager(api_config)
enhanced_predictor = EnhancedTrafficPredictor(traffic_manager, weather_manager)

# Main Title with custom styling
st.markdown('<div class="main-header">🚦 Smart Traffic Prediction System</div>', unsafe_allow_html=True)

# API Status Section
with st.expander("🔧 System Configuration & API Status", expanded=False):
    st.markdown("### TomTom Traffic API")
    st.markdown('<div class="metric-container status-success">✅ API Key Configured<br><small>Real-time traffic flow and incidents</small></div>', unsafe_allow_html=True)
    
    # Test TomTom API
    if st.button("🔍 Test TomTom API", key="test_tomtom"):
        with st.spinner("Testing TomTom API..."):
            test_result = traffic_manager.get_traffic_flow_data(40.7128, -74.0060)  # NYC
            if test_result.get('status') == 'success':
                st.success("✅ TomTom API is working perfectly!")
                with st.expander("API Response Details"):
                    st.json(test_result)
            else:
                st.error(f"❌ TomTom API Error: {test_result.get('error', 'Unknown error')}")

# Sidebar Configuration
st.sidebar.markdown("## 🛠️ Configuration Panel")

# User Input Parameters
st.sidebar.markdown("### 📊 Prediction Parameters")
city = st.sidebar.selectbox("🏙️ Select City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])
day = st.sidebar.selectbox("📅 Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
hour = st.sidebar.slider("🕐 Hour of the Day", 0, 23, 8)
temperature = st.sidebar.slider("🌡️ Temperature (°C)", -10, 40, 22)

st.sidebar.markdown("### 🌤️ Weather Conditions")
rain = st.sidebar.checkbox("🌧️ Rain")
snow = st.sidebar.checkbox("❄️ Snow")
fog = st.sidebar.checkbox("🌫️ Fog")
holiday = st.sidebar.checkbox("🏖️ Holiday")

# Route Analysis Section
st.sidebar.markdown("### 🛣️ Route Analysis")
enable_route_analysis = st.sidebar.checkbox("Enable Route Analysis")
if enable_route_analysis:
    origin_city = st.sidebar.selectbox("📍 Origin City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], key="origin")
    destination_city = st.sidebar.selectbox("📍 Destination City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], key="dest")

# Generate synthetic data and train model
@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)
    data = {
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n),
        'day': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n),
        'hour': np.random.randint(0, 24, n),
        'temperature': np.random.uniform(-10, 40, n),
        'rain': np.random.choice([0, 1], n),
        'snow': np.random.choice([0, 1], n),
        'fog': np.random.choice([0, 1], n),
        'holiday': np.random.choice([0, 1], n)
    }
    df = pd.DataFrame(data)
    df['congestion'] = (
        df['hour'].apply(lambda x: 80 if 7 <= x <= 9 or 17 <= x <= 19 else 30) +
        df['rain'] * 10 + df['snow'] * 15 + df['fog'] * 5 + df['holiday'] * -20 +
        np.random.normal(0, 5, n)
    ).clip(0, 100)
    df['vehicles'] = (df['congestion'] * np.random.uniform(0.8, 1.2, n) * 10).round().astype(int)
    return df

df = generate_data()
df_encoded = pd.get_dummies(df, columns=['city', 'day'], drop_first=True)

@st.cache_resource
def train_model(data):
    X = data.drop(['congestion', 'vehicles'], axis=1)
    y = data['congestion']
    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns

model, feature_names = train_model(df_encoded)

# Prepare input and get predictions
input_data = pd.DataFrame([{
    'hour': hour,
    'temperature': temperature,
    'rain': int(rain),
    'snow': int(snow),
    'fog': int(fog),
    'holiday': int(holiday),
}])

for col in feature_names:
    if col.startswith('city_'):
        input_data[col] = 1 if f"city_{city}" == col else 0
    elif col.startswith('day_'):
        input_data[col] = 1 if f"day_{day}" == col else 0
    elif col not in input_data.columns:
        input_data[col] = 0

base_prediction = model.predict(input_data)[0]
enhanced_result = enhanced_predictor.get_enhanced_prediction(city, base_prediction)

# Main Results Section
st.markdown('<div class="section-header">📊 Traffic Prediction Results</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🤖 Base ML Prediction", f"{base_prediction:.1f}%", help="Machine Learning baseline prediction")
with col2:
    st.metric("🌐 Enhanced Prediction", f"{enhanced_result['prediction']:.1f}%", 
              delta=f"{enhanced_result['adjustment']:.1f}%", help="Real-time data enhanced prediction")
with col3:
    predicted_vehicles = int(enhanced_result['prediction'] * 8)
    st.metric("🚙 Estimated Vehicles", f"{predicted_vehicles:,}", help="Estimated number of vehicles")
with col4:
    if enhanced_result['prediction'] > 70:
        status = "🔴 High Traffic"
    elif enhanced_result['prediction'] > 40:
        status = "🟡 Medium Traffic"
    else:
        status = "🟢 Low Traffic"
    st.metric("🚦 Traffic Status", status, help="Current traffic condition")

# Show adjustment factors
if enhanced_result['factors']:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**🔄 Real-time Adjustments Applied:**")
    for factor in enhanced_result['factors']:
        st.markdown(f"• {factor}")
    st.markdown('</div>', unsafe_allow_html=True)

# Real-time Data Dashboard
st.markdown('<div class="section-header">🌐 Real-time Data Dashboard</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌤️ Weather Conditions")
    weather_data = enhanced_result.get('weather_data', {})
    st.info("🎲 Simulated Weather Data")
    
    weather_col1, weather_col2 = st.columns(2)
    with weather_col1:
        st.metric("Temperature", f"{weather_data.get('temperature', temperature):.1f}°C")
        st.metric("Humidity", f"{weather_data.get('humidity', 50)}%")
    with weather_col2:
        st.metric("Weather", weather_data.get('weather', 'Unknown'))
        if weather_data.get('rain', 0) > 0:
            st.metric("Rain", f"{weather_data['rain']:.1f} mm/h")

with col2:
    st.markdown("### 🚗 Traffic Flow Data")
    flow_data = enhanced_result.get('flow_data', {})
    if flow_data.get('status') == 'success':
        st.success("✅ Live TomTom Data")
        flow_col1, flow_col2 = st.columns(2)
        with flow_col1:
            st.metric("Current Speed", f"{flow_data.get('current_speed', 0):.0f} km/h")
            st.metric("Free Flow Speed", f"{flow_data.get('free_flow_speed', 0):.0f} km/h")
        with flow_col2:
            st.metric("Congestion Level", f"{flow_data.get('congestion_level', 0):.1f}%")
            if flow_data.get('road_closure', False):
                st.error("🚧 Road Closure Detected")
    else:
        st.warning("⚠️ TomTom Data Unavailable")
        st.caption(flow_data.get('error', 'Unknown error'))

with col3:
    st.markdown("### 🚨 Traffic Incidents")
    incidents_data = enhanced_result.get('incidents_data', {})
    if incidents_data.get('status') == 'success':
        incident_count = incidents_data.get('incident_count', 0)
        if incident_count > 0:
            st.error(f"⚠️ {incident_count} Active Incidents")
            for i, incident in enumerate(incidents_data.get('incidents', [])[:2]):
                with st.expander(f"Incident {i+1}: {incident['type']}", expanded=False):
                    st.write(f"**Description:** {incident['description']}")
                    st.write(f"**Magnitude:** {incident['magnitude']}")
                    if incident['start_time']:
                        st.write(f"**Start Time:** {incident['start_time']}")
        else:
            st.success("✅ No Incidents Reported")
    else:
        st.info("ℹ️ Incident Data Unavailable")

# Route Analysis with TomTom
if enable_route_analysis:
    st.markdown('<div class="section-header">🛣️ Route Analysis</div>', unsafe_allow_html=True)
    
    city_coords = {
        'New York': [40.7128, -74.0060],
        'Los Angeles': [34.0522, -118.2437],
        'Chicago': [41.8781, -87.6298],
        'Houston': [29.7604, -95.3698],
        'Phoenix': [33.4484, -112.0740]
    }
    
    if origin_city != destination_city:
        origin_coords = city_coords[origin_city]
        dest_coords = city_coords[destination_city]
        
        with st.spinner("Calculating route with real-time traffic..."):
            route_data = traffic_manager.get_route_data(
                origin_coords[0], origin_coords[1],
                dest_coords[0], dest_coords[1]
            )
        
        if route_data.get('status') == 'success':
            route_col1, route_col2, route_col3, route_col4 = st.columns(4)
            
            with route_col1:
                st.metric("📏 Distance", f"{route_data['distance_km']} km")
            with route_col2:
                st.metric("🚗 With Traffic", f"{route_data['travel_time_minutes']} min")
            with route_col3:
                st.metric("🛣️ Without Traffic", f"{route_data['no_traffic_time_minutes']} min")
            with route_col4:
                st.metric("⏱️ Traffic Delay", f"{route_data['delay_minutes']} min")
            
            if route_data['delay_minutes'] > 5:
                st.warning(f"⚠️ Significant traffic delay: {route_data['delay_minutes']} minutes ({route_data['traffic_delay_percentage']}% increase)")
            else:
                st.success("✅ No significant traffic delays detected")
        else:
            st.error(f"❌ Route calculation failed: {route_data.get('error', 'Unknown error')}")
    else:
        st.info("Please select different origin and destination cities for route analysis")

# Enhanced Map with TomTom data
@st.cache_data
def generate_enhanced_map_data(city_name, congestion_level, flow_data, incidents_data):
    seed_value = hash(city_name) % 1000
    np.random.seed(seed_value)
    
    city_coords = {
        'New York': [40.7128, -74.0060],
        'Los Angeles': [34.0522, -118.2437],
        'Chicago': [41.8781, -87.6298],
        'Houston': [29.7604, -95.3698],
        'Phoenix': [33.4484, -112.0740]
    }
    
    location = city_coords[city_name]
    
    # Adjust total vehicles based on TomTom data
    total_vehicles = int(congestion_level * 8)
    if flow_data.get('status') == 'success':
        flow_congestion = flow_data.get('congestion_level', 0)
        total_vehicles = int((congestion_level + flow_congestion) / 2 * 10)
    
    areas_data = []
    area_names = ["Downtown", "Business District", "Residential Area", "Shopping Center", "Industrial Zone"]
    
    for i in range(5):
        lat_offset = (i - 2) * 0.02
        lon_offset = ((i * 2) % 5 - 2) * 0.02
        
        area_multiplier = [1.5, 1.2, 0.8, 1.0, 0.9][i]
        area_vehicles = int(total_vehicles / 5 * area_multiplier)
        area_congestion = congestion_level * area_multiplier * 0.8
        
        # Add TomTom flow data impact
        if flow_data.get('status') == 'success' and i < 2:  # Apply to main areas
            tomtom_congestion = flow_data.get('congestion_level', 0)
            area_congestion = (area_congestion + tomtom_congestion) / 2
        
        areas_data.append({
            'name': area_names[i],
            'lat': location[0] + lat_offset,
            'lon': location[1] + lon_offset,
            'vehicles': area_vehicles,
            'congestion': min(100, area_congestion),
            'tomtom_enhanced': flow_data.get('status') == 'success' and i < 2
        })
    
    return areas_data, total_vehicles

# Generate enhanced map data
areas_data, total_vehicles = generate_enhanced_map_data(
    city, enhanced_result['prediction'], 
    enhanced_result.get('flow_data', {}),
    enhanced_result.get('incidents_data', {})
)

# Create enhanced map
city_coords = {
    'New York': [40.7128, -74.0060],
    'Los Angeles': [34.0522, -118.2437],
    'Chicago': [41.8781, -87.6298],
    'Houston': [29.7604, -95.3698],
    'Phoenix': [33.4484, -112.0740]
}

location = city_coords[city]
m = folium.Map(location=location, zoom_start=12)

# Add area markers
for area in areas_data:
    if area['congestion'] > 70:
        color = "red"
        status = "High Traffic"
    elif area['congestion'] > 40:
        color = "orange"
        status = "Medium Traffic"
    else:
        color = "green"
        status = "Low Traffic"
    
    enhancement_text = "🌐 Real-time Enhanced" if area.get('tomtom_enhanced', False) else "🤖 ML Predicted"
    
    folium.CircleMarker(
        location=[area['lat'], area['lon']],
        radius=max(8, min(20, area['vehicles'] / 5)),
        popup=f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin: 0; color: #2c3e50;">{area['name']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 2px 0;"><b>🚗 Vehicles:</b> {area['vehicles']:,}</p>
            <p style="margin: 2px 0;"><b>📊 Congestion:</b> {area['congestion']:.1f}%</p>
            <p style="margin: 2px 0;"><b>🚦 Status:</b> {status}</p>
            <p style="margin: 2px 0; font-size: 12px; color: #666;">{enhancement_text}</p>
        </div>
        """,
        tooltip=f"{area['name']}: {area['vehicles']} vehicles ({area['congestion']:.1f}%)",
        color=color,
        fill=True,
        fill_color=color,
        fillOpacity=0.7,
        weight=2
    ).add_to(m)

# Add TomTom incident markers
incidents_data = enhanced_result.get('incidents_data', {})
if incidents_data.get('status') == 'success':
    for i, incident in enumerate(incidents_data.get('incidents', [])[:5]):
        incident_lat = incident.get('lat', location[0])
        incident_lon = incident.get('lon', location[1])
        
        try:
            incident_lat = float(incident_lat)
            incident_lon = float(incident_lon)
        except (ValueError, TypeError):
            incident_lat = location[0] + np.random.uniform(-0.03, 0.03)
            incident_lon = location[1] + np.random.uniform(-0.03, 0.03)
        
        folium.Marker(
            location=[incident_lat, incident_lon],
            popup=f"""
            <div style="font-family: Arial; min-width: 250px;">
                <h4 style="margin: 0; color: #dc3545;">🚨 Traffic Incident</h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 2px 0;"><b>📍 Type:</b> {incident['type']}</p>
                <p style="margin: 2px 0;"><b>📝 Description:</b> {incident['description'][:50]}...</p>
                <p style="margin: 2px 0;"><b>⚠️ Magnitude:</b> {incident['magnitude']}</p>
                <p style="margin: 2px 0;"><b>🕐 Start:</b> {incident.get('start_time', 'Unknown')}</p>
                <p style="margin: 2px 0; font-size: 12px; color: #666;">Source: TomTom API</p>
            </div>
            """,
            tooltip=f"Traffic Incident: {incident['type']}",
            icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
        ).add_to(m)

# Enhanced legend
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 280px; height: 180px; 
     background-color: rgba(255, 255, 255, 0.95); 
     border: 2px solid #333; 
     border-radius: 10px;
     z-index: 9999; 
     font-size: 14px; 
     padding: 15px;
     box-shadow: 0 4px 8px rgba(0,0,0,0.3);
     font-family: Arial, sans-serif;">
<h4 style="margin: 0 0 10px 0; color: #2c3e50; border-bottom: 2px solid #e9ecef; padding-bottom: 5px;">
    🗺️ Traffic Map Legend
</h4>
<div style="margin: 8px 0;">
    <span style="display: inline-block; width: 12px; height: 12px; background-color: #dc3545; border-radius: 50%; margin-right: 8px;"></span>
    <span style="color: #333; font-weight: bold;">High Traffic (>70%)</span>
</div>
<div style="margin: 8px 0;">
    <span style="display: inline-block; width: 12px; height: 12px; background-color: #fd7e14; border-radius: 50%; margin-right: 8px;"></span>
    <span style="color: #333; font-weight: bold;">Medium Traffic (40-70%)</span>
</div>
<div style="margin: 8px 0;">
    <span style="display: inline-block; width: 12px; height: 12px; background-color: #28a745; border-radius: 50%; margin-right: 8px;"></span>
    <span style="color: #333; font-weight: bold;">Low Traffic (<40%)</span>
</div>
<div style="margin: 8px 0;">
    <span style="color: #dc3545; margin-right: 8px;">🚨</span>
    <span style="color: #333; font-weight: bold;">Traffic Incidents</span>
</div>
<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666;">
    <div>• Circle size = Vehicle count</div>
    <div>• Real-time data from TomTom API</div>
</div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Map Section
st.markdown('<div class="section-header">🗺️ Interactive Traffic Map</div>', unsafe_allow_html=True)
st.markdown("**Click on markers for detailed information • Zoom and pan to explore different areas**")
st_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])

# Prediction Summary
st.markdown('<div class="section-header">📋 Prediction Summary</div>', unsafe_allow_html=True)

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown("### 🎯 Current Conditions")
    conditions_data = {
        'Parameter': ['City', 'Day', 'Hour', 'Temperature', 'Weather Conditions'],
        'Value': [
            city,
            day,
            f"{hour}:00",
            f"{temperature}°C",
            f"Rain: {'Yes' if rain else 'No'}, Snow: {'Yes' if snow else 'No'}, Fog: {'Yes' if fog else 'No'}"
        ]
    }
    conditions_df = pd.DataFrame(conditions_data)
    st.dataframe(conditions_df, use_container_width=True, hide_index=True)

with summary_col2:
    st.markdown("### 📊 Prediction Results")
    results_data = {
        'Metric': ['Base ML Prediction', 'Enhanced Prediction', 'Real-time Adjustment', 'Estimated Vehicles', 'Traffic Status'],
        'Value': [
            f"{base_prediction:.1f}%",
            f"{enhanced_result['prediction']:.1f}%",
            f"{enhanced_result['adjustment']:+.1f}%",
            f"{predicted_vehicles:,}",
            status.replace('🔴 ', '').replace('🟡 ', '').replace('🟢 ', '')
        ]
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

# Historical Data Analysis
with st.expander("📈 Historical Data Analysis", expanded=False):
    st.markdown("### Traffic Patterns Analysis")
    
    chart_option = st.selectbox("Select Analysis Type", 
                               ['Hourly Traffic Pattern', 'Daily Traffic Distribution', 'Weather Impact Analysis', 'Vehicle Count Analysis'])
    
    filtered_df = df[df['city'] == city]
    
    if chart_option == 'Hourly Traffic Pattern':
        line_data = filtered_df.groupby('hour')[['congestion', 'vehicles']].mean().reset_index()
        fig = px.line(line_data, x='hour', y=['congestion', 'vehicles'], 
                      title=f'Average Traffic Patterns by Hour - {city}',
                      labels={'value': 'Count/Percentage', 'variable': 'Metric'},
                      color_discrete_map={'congestion': '#ff7f0e', 'vehicles': '#1f77b4'})
        
        # Add current prediction as a point
        fig.add_scatter(x=[hour], y=[enhanced_result['prediction']], 
                       mode='markers', name='Current Enhanced Prediction',
                       marker=dict(size=15, color='red', symbol='star'))
        
    elif chart_option == 'Daily Traffic Distribution':
        fig = px.box(filtered_df, x='day', y='congestion', 
                     title=f'Traffic Congestion Distribution by Day - {city}',
                     color='day')
        
    elif chart_option == 'Weather Impact Analysis':
        weather_impact = filtered_df.groupby(['rain', 'snow', 'fog'])['congestion'].mean().reset_index()
        weather_impact['conditions'] = weather_impact.apply(
            lambda x: f"Rain: {bool(x['rain'])}, Snow: {bool(x['snow'])}, Fog: {bool(x['fog'])}", axis=1
        )
        fig = px.bar(weather_impact, x='conditions', y='congestion',
                     title=f'Weather Impact on Traffic Congestion - {city}',
                     color='congestion', color_continuous_scale='Reds')
        
    else:  # Vehicle Count Analysis
        fig = px.scatter(filtered_df, x='congestion', y='vehicles', 
                         title=f'Vehicle Count vs Congestion Level - {city}',
                         labels={'congestion': 'Congestion (%)', 'vehicles': 'Number of Vehicles'},
                         color='hour', size='temperature')
        # Add current prediction
        fig.add_scatter(x=[enhanced_result['prediction']], y=[total_vehicles], 
                       mode='markers', name='Current Prediction',
                       marker=dict(size=20, color='red', symbol='star'))
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 14px; padding: 20px;">
        <p><strong>Smart Traffic Prediction System</strong> | Powered by Machine Learning & Real-time APIs</p>
        <p>🌐 TomTom Traffic API • 🎲 Simulated Weather • 🤖 Scikit-learn ML</p>
    </div>
    """, 
    unsafe_allow_html=True
)