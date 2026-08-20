import streamlit as st
from supabase import create_client, Client
import base64
from io import BytesIO

def _get_supabase() -> Client:
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

def upload_to_flywheel(image_pil, plant_class_name: str, region: str, season: str, user_id: str = None):
    """Saves a verified image to the community dataset for continuous learning"""
    try:
        supabase = _get_supabase()
        
        # Convert PIL image to base64
        buffered = BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        data = {
            "plant_class_name": plant_class_name,
            "image_base64": img_str,
            "region": region,
            "season": season
        }
        if user_id:
            data["user_id"] = user_id
            
        supabase.table("community_dataset").insert(data).execute()
        return True
    except Exception as e:
        print(f"Error uploading to flywheel: {e}")
        return False

def get_flywheel_data():
    """Fetches community data for automated fine-tuning"""
    try:
        supabase = _get_supabase()
        response = supabase.table("community_dataset").select("*").execute()
        return response.data
    except Exception as e:
        print(f"Error fetching flywheel data: {e}")
        return []
