import os
import streamlit as st
from supabase import create_client, Client

def _get_supabase() -> Client:
    """Helper to initialize Supabase client"""
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

def save_to_garden(user_id: str, plant_name: str, scientific_name: str, health_status: str, notes: str = ""):
    """Saves a plant scan to the user's digital garden"""
    try:
        supabase = _get_supabase()
        data = {
            "user_id": user_id,
            "plant_name": plant_name,
            "scientific_name": scientific_name,
            "health_status": health_status,
            "notes": notes
        }
        supabase.table("digital_garden").insert(data).execute()
        return True
    except Exception as e:
        print(f"Error saving to garden: {e}")
        return False

def get_my_garden(user_id: str):
    """Fetches the user's digital garden"""
    try:
        supabase = _get_supabase()
        response = supabase.table("digital_garden").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"Error fetching garden: {e}")
        return []

def pin_to_foraging_map(user_id: str, user_name: str, plant_name: str, lat: float, lon: float, region: str, climate: str, soil: str):
    """Adds a pin to the public foraging map"""
    try:
        supabase = _get_supabase()
        data = {
            "user_id": user_id,
            "user_name": user_name,
            "plant_name": plant_name,
            "latitude": lat,
            "longitude": lon,
            "region": region,
            "climate": climate,
            "soil": soil
        }
        supabase.table("foraging_map").insert(data).execute()
        return True
    except Exception as e:
        print(f"Error pinning to map: {e}")
        return False

def get_foraging_pins():
    """Fetches all pins for the foraging map"""
    try:
        supabase = _get_supabase()
        response = supabase.table("foraging_map").select("*").execute()
        return response.data
    except Exception as e:
        print(f"Error fetching pins: {e}")
        return []
