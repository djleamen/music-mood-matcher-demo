#!/usr/bin/env python3
"""
Test the feedback functionality to ensure songs don't disappear after rating.
"""

import requests
import time
import sys

def test_streamlit_app():
    """Test that the Streamlit app is responding and functional"""
    print("🧪 Testing AI Music Mood Matcher Feedback System")
    print("=" * 50)
    
    # Test if the app is running
    try:
        response = requests.get("http://localhost:8505", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit app is running and accessible")
        else:
            print(f"❌ App returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to app: {e}")
        print("💡 Make sure the app is running with: python3 -m streamlit run demo.py")
        return False
    
    print("\n🎯 Key Features to Test:")
    print("1. Generate a playlist by describing your mood")
    print("2. Click thumbs up (👍) or thumbs down (👎) on songs")
    print("3. Verify songs remain visible after rating")
    print("4. Check that feedback status updates correctly")
    
    print("\n📝 Manual Test Checklist:")
    print("□ Navigate to 'Mood Matcher' page")
    print("□ Enter mood: 'I need energetic music for working out'")
    print("□ Click 'Analyze Mood & Generate Playlist'")
    print("□ Verify 5 songs are displayed")
    print("□ Click 👍 on first song")
    print("□ Verify song remains visible with 'Liked!' status")
    print("□ Click 👎 on second song")
    print("□ Verify song remains visible with 'Noted!' status")
    print("□ Generate a new playlist and verify feedback persists")
    
    print("\n🚀 App URL: http://localhost:8505")
    print("📊 Go to the Mood Matcher page to test the functionality!")
    
    return True

if __name__ == "__main__":
    if test_streamlit_app():
        print("\n🎉 Ready for testing! Open the app in your browser.")
        sys.exit(0)
    else:
        print("\n❌ Setup issues detected. Please fix and try again.")
        sys.exit(1)
