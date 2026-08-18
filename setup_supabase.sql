-- ============================================================
-- HerbScan AI — Supabase Database Setup
-- Run this in your Supabase SQL Editor (Dashboard → SQL Editor)
-- ============================================================

-- Users table (simple username-based identity)
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Marketplace listings
CREATE TABLE IF NOT EXISTS listings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    seller TEXT NOT NULL REFERENCES users(username),
    title TEXT NOT NULL,
    description TEXT,
    price TEXT,
    listing_type TEXT CHECK (listing_type IN ('sell', 'buy')) NOT NULL,
    plant_category TEXT,
    image_data TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'sold', 'closed')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversations between two users
CREATE TABLE IF NOT EXISTS conversations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user1 TEXT NOT NULL REFERENCES users(username),
    user2 TEXT NOT NULL REFERENCES users(username),
    listing_id UUID REFERENCES listings(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender TEXT NOT NULL REFERENCES users(username),
    text TEXT,
    image_data TEXT,
    sent_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status);
CREATE INDEX IF NOT EXISTS idx_listings_seller ON listings(seller);
CREATE INDEX IF NOT EXISTS idx_conv_user1 ON conversations(user1);
CREATE INDEX IF NOT EXISTS idx_conv_user2 ON conversations(user2);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msg_sent ON messages(sent_at);

-- Enable RLS with permissive policies (anon key access)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE listings ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "public_users" ON users FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "public_listings" ON listings FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "public_conversations" ON conversations FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "public_messages" ON messages FOR ALL USING (true) WITH CHECK (true);
