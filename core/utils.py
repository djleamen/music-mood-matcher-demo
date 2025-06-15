"""
Utility functions for the AI Music Mood Matcher.

This module contains shared utilities to avoid code duplication.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def initialize_openai_client(service_name: str = ""):
    """
    Initialize OpenAI client with consistent error handling.

    Args:
        service_name: Optional service name for logging context

    Returns:
        OpenAI client instance or None if unavailable
    """
    if not OPENAI_AVAILABLE:
        service_msg = f" for {service_name}" if service_name else ""
        print(f"⚠️ OpenAI package not available{service_msg}. Install with: pip install openai")
        return None

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        service_msg = f" for {service_name}" if service_name else ""
        print(f"⚠️ OPENAI_API_KEY not found in environment variables{service_msg}")
        return None

    try:
        client = OpenAI(api_key=api_key)
        service_msg = f" for {service_name}" if service_name else ""
        print(f"✅ OpenAI client initialized successfully{service_msg}")
        return client
    except Exception as e:
        service_msg = f" for {service_name}" if service_name else ""
        print(f"⚠️ Failed to initialize OpenAI client{service_msg}: {e}")
        return None
