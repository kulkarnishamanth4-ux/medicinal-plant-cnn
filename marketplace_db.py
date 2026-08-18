"""
Marketplace & Messaging Database Module (Supabase)
===================================================
Handles all CRUD operations for:
- User identity (username-based)
- P2P plant marketplace listings
- Conversations and real-time messaging
"""

import base64
import io
import streamlit as st
from PIL import Image


# --- SUPABASE CLIENT ---

@st.cache_resource
def _get_supabase():
    """Initialize and cache the Supabase client."""
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        return None


def is_available():
    """Check if Supabase is configured and reachable."""
    return _get_supabase() is not None


# --- IMAGE UTILITIES ---

def image_to_base64(uploaded_file, max_size=(800, 800), quality=80):
    """Compress and encode an uploaded image to base64."""
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_bytes(b64_string):
    """Decode base64 string to bytes for st.image()."""
    return base64.b64decode(b64_string)


# --- USER FUNCTIONS ---

def ensure_user(username):
    """Create user if not exists, return username."""
    sb = _get_supabase()
    if not sb:
        return username
    result = sb.table("users").select("username").eq("username", username).execute()
    if not result.data:
        sb.table("users").insert({
            "username": username,
            "display_name": username
        }).execute()
    return username


# --- LISTING FUNCTIONS ---

def create_listing(seller, title, description, price, listing_type,
                   plant_category="", image_file=None):
    """Create a new marketplace listing."""
    sb = _get_supabase()
    if not sb:
        return None
    data = {
        "seller": seller,
        "title": title,
        "description": description,
        "price": price,
        "listing_type": listing_type,
        "plant_category": plant_category,
        "image_data": image_to_base64(image_file) if image_file else None,
    }
    result = sb.table("listings").insert(data).execute()
    return result.data[0] if result.data else None


def get_active_listings(listing_type=None):
    """Get all active marketplace listings."""
    sb = _get_supabase()
    if not sb:
        return []
    query = sb.table("listings").select("*").eq("status", "active").order(
        "created_at", desc=True
    )
    if listing_type:
        query = query.eq("listing_type", listing_type)
    return query.execute().data or []


def get_listing_by_id(listing_id):
    """Get a single listing by ID."""
    sb = _get_supabase()
    if not sb:
        return None
    result = sb.table("listings").select("*").eq("id", listing_id).execute()
    return result.data[0] if result.data else None


def get_user_listings(username):
    """Get all listings by a specific user."""
    sb = _get_supabase()
    if not sb:
        return []
    return sb.table("listings").select("*").eq("seller", username).order(
        "created_at", desc=True
    ).execute().data or []


def close_listing(listing_id):
    """Mark a listing as closed."""
    sb = _get_supabase()
    if not sb:
        return
    sb.table("listings").update({"status": "closed"}).eq("id", listing_id).execute()


# --- CONVERSATION FUNCTIONS ---

def get_or_create_conversation(current_user, other_user, listing_id=None):
    """Find existing conversation or create a new one."""
    sb = _get_supabase()
    if not sb:
        return None

    # Check direction 1
    q = sb.table("conversations").select("*").eq("user1", current_user).eq(
        "user2", other_user
    )
    if listing_id:
        q = q.eq("listing_id", listing_id)
    result = q.execute()
    if result.data:
        return result.data[0]

    # Check direction 2
    q = sb.table("conversations").select("*").eq("user1", other_user).eq(
        "user2", current_user
    )
    if listing_id:
        q = q.eq("listing_id", listing_id)
    result = q.execute()
    if result.data:
        return result.data[0]

    # Create new conversation
    data = {"user1": current_user, "user2": other_user}
    if listing_id:
        data["listing_id"] = listing_id
    result = sb.table("conversations").insert(data).execute()
    return result.data[0] if result.data else None


def get_user_conversations(username):
    """Get all conversations for a user."""
    sb = _get_supabase()
    if not sb:
        return []
    return sb.table("conversations").select("*").or_(
        f"user1.eq.{username},user2.eq.{username}"
    ).order("created_at", desc=True).execute().data or []


# --- MESSAGE FUNCTIONS ---

def send_message(conversation_id, sender, text=None, image_file=None):
    """Send a message in a conversation."""
    sb = _get_supabase()
    if not sb:
        return None
    data = {
        "conversation_id": conversation_id,
        "sender": sender,
        "text": text,
        "image_data": image_to_base64(image_file, max_size=(600, 600))
        if image_file else None,
    }
    result = sb.table("messages").insert(data).execute()
    return result.data[0] if result.data else None


def get_messages(conversation_id):
    """Get all messages in a conversation, oldest first."""
    sb = _get_supabase()
    if not sb:
        return []
    return sb.table("messages").select("*").eq(
        "conversation_id", conversation_id
    ).order("sent_at").execute().data or []
