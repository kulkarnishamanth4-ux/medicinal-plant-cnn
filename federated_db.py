import os
import streamlit as st
import json
from supabase import create_client, Client

def _get_supabase() -> Client:
    """Helper to initialize Supabase client"""
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

def upload_federated_weights(user_id: str, plant_class_name: str, tensor_weights: str):
    """Uploads a serialized KNN tensor state to the federated database"""
    try:
        supabase = _get_supabase()
        data = {
            "user_id": user_id,
            "plant_class_name": plant_class_name,
            "tensor_weights": tensor_weights
        }
        supabase.table("federated_knowledge").insert(data).execute()
        return True
    except Exception as e:
        print(f"Error uploading federated weights: {e}")
        return False

def get_federated_knowledge():
    """Downloads all global federated weights to sync with the local KNN"""
    try:
        supabase = _get_supabase()
        response = supabase.table("federated_knowledge").select("*").execute()
        return response.data
    except Exception as e:
        print(f"Error fetching federated knowledge: {e}")
        return []
