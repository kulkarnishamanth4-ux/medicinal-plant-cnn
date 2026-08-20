-- ============================================================
-- HerbScan AI — SECURE Supabase Database Setup (Sections 1, 2, 4)
-- Run this in your Supabase SQL Editor (Dashboard → SQL Editor)
-- WARNING: This will DELETE existing marketplace data!
-- ============================================================

-- Clean up old insecure tables if they exist
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS listings;
DROP TABLE IF EXISTS users;

-- Marketplace listings (Now linked securely to auth.users via UUID)
CREATE TABLE IF NOT EXISTS listings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    seller_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    seller_email TEXT NOT NULL, -- Storing email for easy display in UI
    title TEXT NOT NULL,
    description TEXT,
    price TEXT,
    listing_type TEXT CHECK (listing_type IN ('sell', 'buy')) NOT NULL,
    plant_category TEXT,
    image_data TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'sold', 'closed')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversations between two users (Linked securely via UUID)
CREATE TABLE IF NOT EXISTS conversations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user1_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    user2_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    user1_email TEXT NOT NULL,
    user2_email TEXT NOT NULL,
    listing_id UUID REFERENCES listings(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat messages (Linked securely via UUID)
CREATE TABLE IF NOT EXISTS messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    sender_email TEXT NOT NULL,
    text TEXT,
    image_data TEXT,
    sent_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status);
CREATE INDEX IF NOT EXISTS idx_listings_seller ON listings(seller_id);
CREATE INDEX IF NOT EXISTS idx_conv_user1 ON conversations(user1_id);
CREATE INDEX IF NOT EXISTS idx_conv_user2 ON conversations(user2_id);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);

-- ============================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================

ALTER TABLE listings ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- 1. LISTINGS: Anyone can read, but only the owner can insert/update/delete
CREATE POLICY "Public can read active listings" ON listings 
FOR SELECT USING (status = 'active');

CREATE POLICY "Users can manage their own listings" ON listings 
FOR ALL USING (auth.uid() = seller_id) WITH CHECK (auth.uid() = seller_id);

-- 2. CONVERSATIONS: Only participants can read or create
CREATE POLICY "Users can access their own conversations" ON conversations
FOR SELECT USING (auth.uid() = user1_id OR auth.uid() = user2_id);

CREATE POLICY "Users can create conversations" ON conversations
FOR INSERT WITH CHECK (auth.uid() = user1_id OR auth.uid() = user2_id);

-- 3. MESSAGES: Only conversation participants can read, only sender can insert
CREATE POLICY "Participants can read messages in their conversations" ON messages
FOR SELECT USING (
    EXISTS (
        SELECT 1 FROM conversations c 
        WHERE c.id = messages.conversation_id 
        AND (c.user1_id = auth.uid() OR c.user2_id = auth.uid())
    )
);

CREATE POLICY "Users can insert their own messages" ON messages
FOR INSERT WITH CHECK (auth.uid() = sender_id);

-- ============================================================
-- DIGITAL GARDEN & FORAGING MAP
-- ============================================================

DROP TABLE IF EXISTS digital_garden;
DROP TABLE IF EXISTS foraging_map;

CREATE TABLE IF NOT EXISTS digital_garden (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    plant_name TEXT NOT NULL,
    scientific_name TEXT,
    health_status TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS foraging_map (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    user_name TEXT,
    plant_name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    region TEXT,
    climate TEXT,
    soil TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE digital_garden ENABLE ROW LEVEL SECURITY;
ALTER TABLE foraging_map ENABLE ROW LEVEL SECURITY;

-- 4. DIGITAL GARDEN: Only owner can read/insert
CREATE POLICY "Users can manage their own digital garden" ON digital_garden
FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- 5. FORAGING MAP: Public read, owner insert
CREATE POLICY "Public can read foraging map" ON foraging_map
FOR SELECT USING (true);

CREATE POLICY "Users can insert foraging pins" ON foraging_map
FOR INSERT WITH CHECK (auth.uid() = user_id);

-- ============================================================
-- FEDERATED LEARNING
-- ============================================================
DROP TABLE IF EXISTS federated_knowledge;

CREATE TABLE IF NOT EXISTS federated_knowledge (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    plant_class_name TEXT NOT NULL,
    tensor_weights TEXT NOT NULL, -- Stored as JSON string
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE federated_knowledge ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public can read federated knowledge" ON federated_knowledge
FOR SELECT USING (true);

CREATE POLICY "Users can contribute to federated knowledge" ON federated_knowledge
FOR INSERT WITH CHECK (auth.uid() = user_id);
