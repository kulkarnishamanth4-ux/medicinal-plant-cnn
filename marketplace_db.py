"""
Marketplace & Messaging Database Module (Supabase)
===================================================
Handles all CRUD operations securely using Supabase Auth and RLS.
"""

import base64
import io
import streamlit as st
from PIL import Image


# --- SUPABASE CLIENT (SESSION AWARE) ---

def get_client():
    """Create a new Supabase client for the current Streamlit session."""
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        client = create_client(url, key)
        
        # Restore the user's auth session if they are logged in
        if "sb_access_token" in st.session_state and "sb_refresh_token" in st.session_state:
            client.auth.set_session(
                st.session_state["sb_access_token"], 
                st.session_state["sb_refresh_token"]
            )
        return client
    except Exception as e:
        st.error(f"Supabase Init Error: {e}")
        return None

def is_available():
    return get_client() is not None


# --- IMAGE UTILITIES ---

def image_to_base64(uploaded_file, max_size=(800, 800), quality=80):
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_bytes(b64_string):
    return base64.b64decode(b64_string)


# --- AUTHENTICATION ---

def sign_up(email, password):
    sb = get_client()
    if not sb: return False, "Database unavailable"
    try:
        res = sb.auth.sign_up({"email": email, "password": password})
        if res.user:
            return True, "Success! You can now log in."
        return False, "Sign up failed."
    except Exception as e:
        return False, str(e)

def login(email, password):
    sb = get_client()
    if not sb: return False, "Database unavailable"
    try:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        if res.session:
            st.session_state["sb_access_token"] = res.session.access_token
            st.session_state["sb_refresh_token"] = res.session.refresh_token
            st.session_state["user_id"] = res.user.id
            st.session_state["user_email"] = res.user.email
            return True, "Logged in!"
        return False, "Invalid credentials."
    except Exception as e:
        return False, str(e)

def logout():
    sb = get_client()
    if sb:
        try: sb.auth.sign_out()
        except: pass
    st.session_state.pop("sb_access_token", None)
    st.session_state.pop("sb_refresh_token", None)
    st.session_state.pop("user_id", None)
    st.session_state.pop("user_email", None)


# --- LISTING FUNCTIONS ---

def create_listing(title, description, price, listing_type, plant_category="", image_file=None):
    sb = get_client()
    if not sb or "user_id" not in st.session_state: return None
    data = {
        "seller_id": st.session_state["user_id"],
        "seller_email": st.session_state["user_email"],
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
    sb = get_client()
    if not sb: return []
    query = sb.table("listings").select("*").eq("status", "active").order("created_at", desc=True)
    if listing_type:
        query = query.eq("listing_type", listing_type)
    return query.execute().data or []

def get_listing_by_id(listing_id):
    sb = get_client()
    if not sb: return None
    result = sb.table("listings").select("*").eq("id", listing_id).execute()
    return result.data[0] if result.data else None

def get_user_listings():
    sb = get_client()
    if not sb or "user_id" not in st.session_state: return []
    return sb.table("listings").select("*").eq("seller_id", st.session_state["user_id"]).order("created_at", desc=True).execute().data or []

def close_listing(listing_id):
    sb = get_client()
    if not sb or "user_id" not in st.session_state: return False
    # Ensure the user only closes their own listing
    try:
        sb.table("listings").update({"status": "closed"}).eq("id", listing_id).eq("seller_id", st.session_state["user_id"]).execute()
        return True
    except Exception as e:
        print(f"Error closing listing: {e}")
        return False


# --- CONVERSATION FUNCTIONS ---

def get_or_create_conversation(other_user_id, other_user_email, listing_id=None):
    sb = get_client()
    if not sb or "user_id" not in st.session_state: return None
    my_id = st.session_state["user_id"]
    my_email = st.session_state["user_email"]

    # Check direction 1
    q = sb.table("conversations").select("*").eq("user1_id", my_id).eq("user2_id", other_user_id)
    if listing_id: q = q.eq("listing_id", listing_id)
    result = q.execute()
    if result.data: return result.data[0]

    # Check direction 2
    q = sb.table("conversations").select("*").eq("user1_id", other_user_id).eq("user2_id", my_id)
    if listing_id: q = q.eq("listing_id", listing_id)
    result = q.execute()
    if result.data: return result.data[0]

    # Create new conversation
    data = {
        "user1_id": my_id, 
        "user2_id": other_user_id,
        "user1_email": my_email,
        "user2_email": other_user_email
    }
    if listing_id:
        data["listing_id"] = listing_id
    result = sb.table("conversations").insert(data).execute()
    return result.data[0] if result.data else None

def get_user_conversations():
    sb = get_client()
    if not sb or "user_id" not in st.session_state: return []
    my_id = st.session_state["user_id"]
    return sb.table("conversations").select("*").or_(
        f"user1_id.eq.{my_id},user2_id.eq.{my_id}"
    ).order("created_at", desc=True).execute().data or []


# --- MESSAGE FUNCTIONS ---

def send_message(conversation_id, text=None, image_file=None):
    sb = get_client()
    if not sb or "user_id" not in st.session_state: return None
    data = {
        "conversation_id": conversation_id,
        "sender_id": st.session_state["user_id"],
        "sender_email": st.session_state["user_email"],
        "text": text,
        "image_data": image_to_base64(image_file, max_size=(600, 600)) if image_file else None,
    }
    result = sb.table("messages").insert(data).execute()
    return result.data[0] if result.data else None

def get_messages(conversation_id):
    sb = get_client()
    if not sb: return []
    return sb.table("messages").select("*").eq("conversation_id", conversation_id).order("sent_at").execute().data or []

# --- BLOCKCHAIN (PROVENANCE) ---

def hash_transaction(listing_id, buyer_email, seller_email, price):
    import hashlib
    import datetime
    
    # Create a deterministic string to hash
    tx_string = f"TXN|{listing_id}|{buyer_email}|{seller_email}|{price}|{datetime.date.today().isoformat()}"
    return hashlib.sha256(tx_string.encode('utf-8')).hexdigest()
