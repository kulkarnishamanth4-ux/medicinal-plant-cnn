import json
import os
import base64
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from deficiency_analyzer import analyze_leaf_health
from community_manager import (
    add_unknown_plant, submit_identification,
    get_unidentified_plants, get_all_submissions
)
import marketplace_db as mdb

MODEL_PATH = "medicinal_leaf_efficientnet.keras"
CLASS_NAMES_PATH = "class_names.json"
PLANT_DB_PATH = "plant_database.json"
DEFICIENCY_DB_PATH = "deficiency_database.json"
UPLOADS_DIR = "uploads"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 60.0

os.makedirs(UPLOADS_DIR, exist_ok=True)

st.set_page_config(
    page_title="HerbScan AI - Plant & Health Diagnostic",
    page_icon="🌿", layout="wide", initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #f3f8f5; }
.block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 3rem; }
header, [data-testid="stToolbar"], #MainMenu, .stDeployButton { visibility: hidden; }
.hero {
    background: linear-gradient(145deg, #0f3b26 0%, #1d6b3f 55%, #2d8a53 100%);
    border-radius: 28px; padding: 2.2rem 2.8rem; color: white;
    margin-bottom: 2rem; box-shadow: 0 18px 36px rgba(22,70,40,0.22);
    position: relative; overflow: hidden;
}
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.2);
    padding: 0.35rem 1.1rem; border-radius: 60px; font-size: 0.78rem;
    letter-spacing: 0.6px; font-weight: 700; text-transform: uppercase; margin-bottom: 0.8rem;
}
.hero-title { font-size: 3rem; font-weight: 800; letter-spacing: -1.2px; line-height: 1.1; }
.hero-sub { font-size: 1.05rem; opacity: 0.88; max-width: 700px; margin-top: 0.6rem; line-height: 1.5; }
.card-box {
    background: white; border-radius: 24px; padding: 1.6rem 1.8rem;
    box-shadow: 0 10px 24px rgba(20,60,35,0.05); border: 1px solid #e0ece3; margin-bottom: 1.5rem;
}
.pred-name { font-size: 2.4rem; font-weight: 800; color: #124a29; line-height: 1.1; margin: 0.2rem 0; }
.scientific { font-style: italic; color: #52705e; font-size: 1rem; font-weight: 600; }
.tag { background: #eaf4ed; color: #1c5f37; border: 1px solid #cde2d5; padding: 0.25rem 0.9rem;
    border-radius: 40px; font-size: 0.82rem; font-weight: 600; margin-right: 6px; margin-bottom: 6px; display: inline-block; }
.bullet-item { padding: 0.3rem 0; color: #2b4d3a; font-size: 0.95rem; line-height: 1.5; }
.bullet-item::before { content: "• "; color: #2d8a53; font-weight: 800; font-size: 1.1rem; }
.community-card {
    background: white; border-radius: 20px; padding: 1.2rem 1.5rem;
    border: 1px solid #e2eee6; box-shadow: 0 6px 16px rgba(20,60,35,0.04);
    margin-bottom: 1rem;
}
.footer { text-align: center; color: #7a9583; font-size: 0.82rem; padding-top: 1.8rem;
    border-top: 1px solid #deece4; margin-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# --- CACHED LOADERS ---
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        try: return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e: st.error(f"Model error: {e}"); return None
    return None

@st.cache_data
def get_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return {}

model = get_model()
class_names = get_json(CLASS_NAMES_PATH)
plant_db = get_json(PLANT_DB_PATH)
deficiency_db = get_json(DEFICIENCY_DB_PATH)

if model is None or not class_names:
    st.error("Model or class_names.json missing. Upload them to continue.")
    st.stop()

# --- HERO ---
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ AI Plant Classification & Community Identification</div>
    <div class="hero-title">HerbScan AI 🌿</div>
    <div class="hero-sub">Identify medicinal plants, analyze leaf health deficiencies, and help the community identify unknown species.</div>
</div>
""", unsafe_allow_html=True)

def render_auth_ui():
    st.markdown("### 🔒 Secure Authentication Required")
    st.info("You must log in with a secure account to access the Marketplace and Messages.")
    
    auth_mode = st.radio("Choose action", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
    with st.form("auth_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button(auth_mode, use_container_width=True)
        
        if submit:
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                if auth_mode == "Login":
                    success, msg = mdb.login(email, password)
                else:
                    success, msg = mdb.sign_up(email, password)
                
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

# --- NAVIGATION TABS ---
main_tab, market_tab, msg_tab, gallery_tab = st.tabs([
    "🔬 Scan & Analyze",
    "🛒 Marketplace",
    "💬 Messages",
    "🌍 Community Gallery"
])

# ============================================================
# TAB 1: SCAN & ANALYZE
# ============================================================
with main_tab:
    st.markdown("### 📷 Capture or Upload Leaf Image")
    input_mode = st.radio("Input method", ["📁 Upload File", "📸 Camera"], horizontal=True, label_visibility="collapsed")
    uploaded_file = None
    if input_mode == "📁 Upload File":
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    else:
        cam_photo = st.camera_input("Take a photo of the leaf")
        if cam_photo is not None:
            uploaded_file = cam_photo

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        resized_img = image.resize(IMG_SIZE)
        img_array = np.expand_dims(np.asarray(resized_img, dtype=np.float32), axis=0)
        preds = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        predicted_species = class_names[top_idx]
        species_confidence = float(preds[top_idx]) * 100
        health_results = analyze_leaf_health(image)
        plant_info = plant_db.get(predicted_species, {})
        deficiency_info = deficiency_db.get(health_results["primary_deficiency"], deficiency_db.get("Healthy", {}))

        is_low_confidence = species_confidence < CONFIDENCE_THRESHOLD

        # --- LOW CONFIDENCE: UNKNOWN PLANT FLOW ---
        if is_low_confidence:
            st.warning(f"⚠️ **Low confidence prediction: {species_confidence:.1f}%** — This plant may not be in our database.")
            col_img, col_action = st.columns([1, 1.2], gap="large")

            with col_img:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.caption(f"Model's best guess: **{predicted_species}** ({species_confidence:.1f}%)")

            with col_action:
                st.markdown("### 🤔 We're not sure about this plant")
                st.markdown("Choose an option below:")

                option = st.radio("What would you like to do?", [
                    "🏷️ I know this plant — let me provide the name",
                    "❓ I don't know either — post to Community Gallery for help"
                ], label_visibility="collapsed")

                if option.startswith("🏷"):
                    with st.form("name_form"):
                        user_name = st.text_input("Your name (optional)", placeholder="Anonymous")
                        plant_name = st.text_input("Plant name *", placeholder="e.g. Tulsi, Ashwagandha...")
                        submitted = st.form_submit_button("✅ Submit Plant Name", use_container_width=True)
                        if submitted and plant_name.strip():
                            fname = f"{plant_name.strip().replace(' ','_')}_{uploaded_file.name}"
                            fpath = os.path.join(UPLOADS_DIR, fname)
                            image.save(fpath)
                            sid = add_unknown_plant(fname, predicted_species, species_confidence, user_name, plant_name.strip())
                            st.success(f"✅ Thank you! **{plant_name}** has been recorded. (ID: {sid})")
                        elif submitted:
                            st.error("Please enter a plant name.")

                else:
                    with st.form("unknown_form"):
                        user_name = st.text_input("Your name (optional)", placeholder="Anonymous")
                        submitted = st.form_submit_button("📤 Post to Community Gallery", use_container_width=True)
                        if submitted:
                            fname = f"unknown_{uploaded_file.name}"
                            fpath = os.path.join(UPLOADS_DIR, fname)
                            image.save(fpath)
                            sid = add_unknown_plant(fname, predicted_species, species_confidence, user_name, "")
                            st.success(f"📤 Posted to Community Gallery! Others can now help identify it. (ID: {sid})")

        # --- HIGH CONFIDENCE: NORMAL IDENTIFICATION ---
        else:
            id_tab, health_tab = st.tabs(["🌿 Plant Species & Profile", "🔬 Leaf Health & Deficiency"])

            with id_tab:
                c1, c2 = st.columns([1, 1.2], gap="large")
                with c1:
                    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
                with c2:
                    st.markdown(f"""
                    <div class="card-box">
                        <div style="font-size:0.75rem;font-weight:700;color:#5f7d6b;text-transform:uppercase;">Predicted Species</div>
                        <div class="pred-name">{predicted_species}</div>
                        <div class="scientific">{plant_info.get("scientific_name", plant_info.get("scientific",""))}</div>
                        <div style="margin-top:1rem;font-weight:700;color:#184c2e;">Confidence: {species_confidence:.1f}%</div>
                        <div style="background:#e4f0e8;border-radius:30px;height:10px;width:100%;margin-top:4px;">
                            <div style="background:linear-gradient(90deg,#2d8a53,#51a675);width:{min(species_confidence,100)}%;height:100%;border-radius:30px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    top3 = np.argsort(preds)[-3:][::-1]
                    for rank, idx in enumerate(top3, 1):
                        st.write(f"**{rank}. {class_names[idx]}**: {preds[idx]*100:.1f}%")

                if plant_info:
                    st.markdown("---")
                    st.markdown(f"### 📖 About {predicted_species}")
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.info(f"**Family:**\n{plant_info.get('family','N/A')}")
                    pc2.info(f"**Common Names:**\n{', '.join(plant_info.get('common_names',[predicted_species]))}")
                    pc3.info(f"**Sanskrit:**\n{', '.join(plant_info.get('sanskrit_names',['N/A']))}")
                    if "ayurvedic_properties" in plant_info:
                        st.markdown("#### 🧘 Ayurvedic Classification")
                        ay = plant_info["ayurvedic_properties"]
                        a1,a2,a3,a4 = st.columns(4)
                        a1.metric("Rasa", ay.get("rasa","N/A"))
                        a2.metric("Guna", ay.get("guna","N/A"))
                        a3.metric("Virya", ay.get("virya","N/A"))
                        a4.metric("Vipaka", ay.get("vipaka","N/A"))
                    st.markdown("#### 🧪 Benefits & Constituents")
                    bc1,bc2 = st.columns(2)
                    with bc1:
                        for b in plant_info.get("benefits", plant_info.get("uses",[])):
                            st.markdown(f"<div class='bullet-item'>{b}</div>", unsafe_allow_html=True)
                    with bc2:
                        for c in plant_info.get("active_constituents", plant_info.get("characteristics",[])):
                            st.markdown(f"<span class='tag'>{c}</span>", unsafe_allow_html=True)
                    if "precautions" in plant_info:
                        st.warning(f"⚠️ {plant_info['precautions']}")

            with health_tab:
                hc1, hc2 = st.columns([1, 1.2], gap="large")
                with hc1:
                    st.image(health_results["diagnostic_heatmap"], caption="Diagnostic Heatmap (🟢Healthy 🟡Chlorosis 🟤Scorch 🟣Purpling)", use_container_width=True)
                with hc2:
                    hs = health_results["health_score"]
                    st.markdown(f"""
                    <div class="card-box">
                        <div style="font-size:0.75rem;font-weight:700;color:#5f7d6b;text-transform:uppercase;">Leaf Health Score</div>
                        <div style="font-size:1.8rem;font-weight:800;color:{'#1b6838' if hs>=75 else '#8a6008'};">{hs}%</div>
                        <div style="font-size:0.75rem;font-weight:700;color:#5f7d6b;text-transform:uppercase;margin-top:0.8rem;">Detected Condition</div>
                        <div style="font-size:1.4rem;font-weight:800;color:#8a2e1d;">{health_results['primary_deficiency']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    m = health_results["metrics"]
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Green", f"{m['green_pct']}%")
                    m2.metric("Chlorosis", f"{m['chlorosis_pct']}%")
                    m3.metric("Scorch", f"{m['necrosis_pct']}%")
                    m4.metric("Purple", f"{m['purpling_pct']}%")
                st.markdown("---")
                st.markdown(f"### 🔬 {deficiency_info.get('title', health_results['primary_deficiency'])}")
                st.info(deficiency_info.get("description","Analysis completed."))
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("#### 👁 Symptoms")
                    for s in health_results["symptoms"] + deficiency_info.get("symptoms",[]):
                        st.markdown(f"<div class='bullet-item'>{s}</div>", unsafe_allow_html=True)
                with rc2:
                    st.markdown("#### 🌱 Organic Remedies")
                    for r in deficiency_info.get("organic_remedies",[]):
                        st.markdown(f"<div class='bullet-item'>{r}</div>", unsafe_allow_html=True)
                if "preventive_care" in deficiency_info:
                    st.success(f"💡 **Prevention:** {deficiency_info['preventive_care']}")
    else:
        st.markdown("""
        <div style="background:white;border-radius:24px;padding:3rem;text-align:center;border:1px solid #dce8e0;">
            <div style="font-size:3.5rem;">🍃</div>
            <h3 style="color:#1f4a30;">Ready to Identify & Analyze a Leaf?</h3>
            <p style="color:#5f7d6b;">Upload a clear leaf image above for species classification, health diagnostics, and community identification.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2: MARKETPLACE
# ============================================================
with market_tab:
    if not mdb.is_available():
        st.warning("⚠️ **Marketplace requires Supabase.** Add your Supabase `url` and `key` to `.streamlit/secrets.toml` to enable this feature.")
    else:
        # --- Authentication Session ---
        if "sb_access_token" not in st.session_state:
            render_auth_ui()
        else:
            me = st.session_state.get("user_email", "User")
            c1, c2 = st.columns([4, 1])
            c1.caption(f"Logged in as **{me}**")
            if c2.button("Logout", key="logout_mp"):
                mdb.logout()
                st.rerun()

            mp_view = st.radio("View", ["🛍️ Browse Listings", "➕ Create Listing", "📦 My Listings"], horizontal=True, label_visibility="collapsed")

            # --- Browse Listings ---
            if mp_view == "🛍️ Browse Listings":
                st.markdown("### 🛍️ Active Marketplace Listings")
                filt = st.selectbox("Filter", ["All", "Selling", "Buying"], label_visibility="collapsed")
                lt = {"Selling": "sell", "Buying": "buy"}.get(filt)
                listings = mdb.get_active_listings(listing_type=lt)
                if not listings:
                    st.info("No active listings yet. Be the first to post!")
                else:
                    for i in range(0, len(listings), 2):
                        cols = st.columns(2, gap="medium")
                        for ci, col in enumerate(cols):
                            li = i + ci
                            if li >= len(listings):
                                break
                            item = listings[li]
                            with col:
                                st.markdown(f"""
                                <div class="card-box">
                                    <span class="tag">{"🏷️ Selling" if item["listing_type"]=="sell" else "🔎 Buying"}</span>
                                    {"<span class='tag'>"+item["plant_category"]+"</span>" if item.get("plant_category") else ""}
                                    <div class="pred-name" style="font-size:1.4rem;margin:0.5rem 0;">{item["title"]}</div>
                                    <div style="color:#52705e;font-size:0.9rem;">{item.get("description","")[:120]}</div>
                                    <div style="margin-top:0.6rem;font-weight:700;color:#1d6b3f;font-size:1.1rem;">{item.get("price","—")}</div>
                                    <div style="margin-top:0.4rem;font-size:0.78rem;color:#7a9583;">by {item["seller_email"]} · {item["created_at"][:10]}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                if item.get("image_data"):
                                    st.image(mdb.base64_to_bytes(item["image_data"]), use_container_width=True)
                                if item["seller_id"] != st.session_state.get("user_id"):
                                    if st.button(f"💬 Message Seller", key=f"msg_{item['id']}"):
                                        conv = mdb.get_or_create_conversation(item["seller_id"], item["seller_email"], item["id"])
                                        if conv:
                                            st.session_state.active_conv = conv["id"]
                                            st.session_state.active_conv_other = item["seller_email"]
                                            st.info(f"Conversation opened! Switch to the **💬 Messages** tab to chat.")

            # --- Create Listing ---
            elif mp_view == "➕ Create Listing":
                st.markdown("### ➕ Create a New Listing")
                with st.form("new_listing"):
                    title = st.text_input("Title *", placeholder="e.g. Fresh Tulsi Cuttings")
                    desc = st.text_area("Description", placeholder="Describe what you're selling or looking for...")
                    lc1, lc2, lc3 = st.columns(3)
                    price = lc1.text_input("Price", placeholder="e.g. ₹150 / Free")
                    l_type = lc2.selectbox("Type", ["sell", "buy"])
                    category = lc3.text_input("Plant Category", placeholder="e.g. Tulsi")
                    img = st.file_uploader("Listing image (optional)", type=["jpg","jpeg","png","webp"])
                    if st.form_submit_button("📤 Post Listing", use_container_width=True):
                        if title.strip():
                            result = mdb.create_listing(title.strip(), desc, price, l_type, category, img)
                            if result:
                                st.success("✅ Listing created successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to create listing.")
                        else:
                            st.error("Please enter a title.")

            # --- My Listings ---
            else:
                st.markdown("### 📦 My Listings")
                my_items = mdb.get_user_listings()
                if not my_items:
                    st.info("You haven't posted any listings yet.")
                else:
                    for item in my_items:
                        with st.container():
                            st.markdown(f"""
                            <div class="community-card" style="border-left:4px solid {'#2d8a53' if item['status']=='active' else '#999'};">
                                <span class="tag">{item["listing_type"].upper()}</span>
                                <span class="tag">{item["status"].upper()}</span>
                                <div style="font-weight:700;font-size:1.1rem;margin:0.3rem 0;">{item["title"]}</div>
                                <div style="font-size:0.85rem;color:#52705e;">{item.get("price","—")} · {item["created_at"][:10]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            if item["status"] == "active":
                                if st.button("❌ Close Listing", key=f"close_{item['id']}"):
                                    mdb.close_listing(item["id"])
                                    st.success("Listing closed.")
                                    st.rerun()

# ============================================================
# TAB 3: MESSAGES
# ============================================================
with msg_tab:
    if not mdb.is_available():
        st.warning("⚠️ **Messaging requires Supabase.** Add your Supabase `url` and `key` to `.streamlit/secrets.toml` to enable this feature.")
    else:
        if "sb_access_token" not in st.session_state:
            st.info("Please log in using the **🛒 Marketplace** tab first.")
        else:
            me = st.session_state.get("user_email", "")
            convos = mdb.get_user_conversations()

            if "active_conv" not in st.session_state:
                st.session_state.active_conv = None
            if "active_conv_other" not in st.session_state:
                st.session_state.active_conv_other = ""

            mc1, mc2 = st.columns([1, 2.5], gap="medium")

            with mc1:
                st.markdown("#### 📬 Conversations")
                if not convos:
                    st.caption("No conversations yet. Message a seller from the Marketplace!")
                for c in convos:
                    other = c["user2_email"] if c["user1_id"] == st.session_state.get("user_id") else c["user1_email"]
                    label = f"💬 {other}"
                    if c.get("listing_id"):
                        label += " (listing)"
                    if st.button(label, key=f"conv_{c['id']}", use_container_width=True):
                        st.session_state.active_conv = c["id"]
                        st.session_state.active_conv_other = other
                        st.rerun()

            with mc2:
                if st.session_state.active_conv:
                    other = st.session_state.active_conv_other
                    st.markdown(f"#### Chat with **{other}**")
                    messages = mdb.get_messages(st.session_state.active_conv)

                    chat_container = st.container(height=400)
                    with chat_container:
                        if not messages:
                            st.caption("No messages yet. Say hello! 👋")
                        for m in messages:
                            is_me = m["sender"] == me
                            align = "right" if is_me else "left"
                            bg = "#e8f5e9" if is_me else "#ffffff"
                            border_col = "#2d8a53" if is_me else "#dce8e0"
                            st.markdown(f"""
                            <div style="text-align:{align};margin:0.3rem 0;">
                                <div style="display:inline-block;background:{bg};border:1px solid {border_col};
                                    border-radius:16px;padding:0.6rem 1rem;max-width:75%;text-align:left;">
                                    <div style="font-size:0.7rem;font-weight:700;color:#5f7d6b;">{m["sender_email"]}</div>
                                    {"<div style='font-size:0.92rem;color:#1f3a2a;'>"+m["text"]+"</div>" if m.get("text") else ""}
                                    <div style="font-size:0.65rem;color:#9ab3a1;margin-top:2px;">{m["sent_at"][:16] if m.get("sent_at") else ""}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            if m.get("image_data"):
                                st.image(mdb.base64_to_bytes(m["image_data"]), width=250)

                    with st.form("send_msg", clear_on_submit=True):
                        sm1, sm2 = st.columns([4, 1])
                        msg_text = sm1.text_input("Message", placeholder="Type a message...", label_visibility="collapsed")
                        msg_img = st.file_uploader("Attach image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
                        if sm2.form_submit_button("Send ➤"):
                            if msg_text.strip() or msg_img:
                                mdb.send_message(st.session_state.active_conv, msg_text.strip() or None, msg_img)
                                st.rerun()
                            else:
                                st.warning("Enter a message or attach an image.")
                else:
                    st.markdown("""
                    <div style="background:white;border-radius:20px;padding:3rem;text-align:center;border:1px solid #dce8e0;">
                        <div style="font-size:3rem;">💬</div>
                        <h3 style="color:#1f4a30;">Select a Conversation</h3>
                        <p style="color:#5f7d6b;">Choose a conversation from the list, or start one from the Marketplace.</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================
# TAB 4: COMMUNITY GALLERY
# ============================================================
with gallery_tab:
    st.markdown("### 🌍 Community Plant Gallery")
    st.markdown("These plants were uploaded by users but couldn't be confidently identified. **Help the community by suggesting a name!**")
    unidentified = get_unidentified_plants()
    all_subs = get_all_submissions()

    if not all_subs:
        st.info("No community submissions yet. Upload a leaf image that the model can't identify to get started!")
    else:
        # Show unidentified plants first
        if unidentified:
            st.markdown(f"#### ❓ Awaiting Identification ({len(unidentified)} plants)")
            for plant in unidentified:
                with st.container():
                    st.markdown(f'<div class="community-card">', unsafe_allow_html=True)
                    gc1, gc2 = st.columns([1, 2])
                    with gc1:
                        img_path = os.path.join(UPLOADS_DIR, plant["image_filename"])
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.write("📷 Image not available")
                    with gc2:
                        st.markdown(f"**Submission ID:** `{plant['id']}`")
                        st.markdown(f"**Model's guess:** {plant['model_prediction']} ({plant['model_confidence']}%)")
                        st.markdown(f"**Submitted by:** {plant['submitted_by']} on {plant['submitted_at']}")
                        if plant["identification_suggestions"]:
                            st.markdown("**💡 Community Suggestions:**")
                            for sug in plant["identification_suggestions"]:
                                st.write(f"- **{sug['suggested_name']}** (by {sug['suggested_by']})")
                        with st.form(f"suggest_{plant['id']}"):
                            sc1, sc2 = st.columns(2)
                            sug_name = sc1.text_input("Suggested plant name", key=f"sn_{plant['id']}", placeholder="e.g. Tulsi")
                            sug_by = sc2.text_input("Your name", key=f"sb_{plant['id']}", placeholder="Anonymous")
                            if st.form_submit_button("📩 Submit Identification", use_container_width=True):
                                if sug_name.strip():
                                    submit_identification(plant["id"], sug_name.strip(), sug_by)
                                    st.success(f"Thank you! Your suggestion '{sug_name}' has been recorded.")
                                    st.rerun()
                                else:
                                    st.error("Please enter a plant name.")
                    st.markdown("</div>", unsafe_allow_html=True)

        # Show identified submissions
        identified = [p for p in all_subs if p["status"] == "identified"]
        if identified:
            st.markdown(f"#### ✅ User-Identified Plants ({len(identified)})")
            for plant in identified:
                st.markdown(f'<div class="community-card" style="border-left:4px solid #2d8a53;">', unsafe_allow_html=True)
                ic1, ic2 = st.columns([1, 3])
                with ic1:
                    img_path = os.path.join(UPLOADS_DIR, plant["image_filename"])
                    if os.path.exists(img_path):
                        st.image(img_path, width=150)
                with ic2:
                    st.markdown(f"**🏷️ {plant['user_provided_name']}** — identified by {plant['submitted_by']}")
                    st.caption(f"Model guess: {plant['model_prediction']} ({plant['model_confidence']}%) | {plant['submitted_at']}")
                st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    HerbScan AI · Plant ID · Health Diagnostics · Marketplace · Messaging · Community Gallery<br>
    Built with TensorFlow + EfficientNetB0 + OpenCV + Supabase + Streamlit
</div>
""", unsafe_allow_html=True)
