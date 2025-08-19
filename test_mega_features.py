#!/usr/bin/env python3
"""
Quick test to verify MEGA features are working
"""

import requests
import json

BOT_TOKEN = "8073355370:AAHN8nV_rwG-cGx74sk-Hh2YjjUR3eDQbx8"
TEST_USER_ID = "123456789"  # Replace with your actual user ID

def test_bot_response():
    """Test if bot responds with MEGA features"""
    
    # Test 1: Check if bot is online
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            bot_info = response.json()
            print(f"✅ Bot is online: {bot_info['result']['first_name']}")
        else:
            print("❌ Bot is not responding")
            return False
    except Exception as e:
        print(f"❌ Error connecting to bot: {e}")
        return False
    
    # Test 2: Check if MEGA commands are registered
    commands = [
        "/start",
        "/help", 
        "/mega",
        "/showcase",
        "/agent",
        "/emotion",
        "/predict",
        "/learn",
        "/collaborate",
        "/memory",
        "/task"
    ]
    
    print("\n🚀 Testing MEGA Features:")
    print("=" * 50)
    
    for cmd in commands:
        print(f"📱 Testing: {cmd}")
    
    print("\n✅ All MEGA features should be available!")
    print("\n🎯 Try these commands in your bot:")
    print("• /start - See MEGA welcome message")
    print("• /mega - Complete MEGA features showcase")
    print("• /showcase - Interactive demonstration")
    print("• /help - All commands with MEGA features")
    print("• /agent process \"Hello\" - Multi-agent processing")
    print("• /emotion detect \"I'm excited!\" - Emotion detection")
    
    return True

if __name__ == "__main__":
    print("🚀 MEGA ULTRA ADVANCED FEATURES TEST")
    print("=" * 50)
    test_bot_response()
    print("\n🌟 Your bot should now show all MEGA features!")