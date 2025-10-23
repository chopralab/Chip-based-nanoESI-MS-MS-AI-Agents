"""
Configuration Loader for API Keys
==================================
Safely loads API keys from environment variables or .env file.
"""

import os
from pathlib import Path


def load_api_key(env_file_path: str = None) -> str:
    """
    Load OpenAI API key from environment or .env file.
    
    Priority:
    1. Already set in os.environ
    2. .env file in specified path
    3. .env file in helper_agent directory
    4. System environment variable
    
    Args:
        env_file_path: Optional path to .env file
    
    Returns:
        API key string
    
    Raises:
        ValueError: If API key not found
    """
    # Check if already set
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    
    # Try to load from .env file using python-dotenv
    try:
        from dotenv import load_dotenv
        
        # Try specified path
        if env_file_path:
            env_path = Path(env_file_path)
            if env_path.exists():
                load_dotenv(env_path)
                if os.environ.get("OPENAI_API_KEY"):
                    print(f"✅ Loaded API key from: {env_path}")
                    return os.environ["OPENAI_API_KEY"]
        
        # Try default location (helper_agent/.env)
        default_env = Path(__file__).parent.parent / ".env"
        if default_env.exists():
            load_dotenv(default_env)
            if os.environ.get("OPENAI_API_KEY"):
                print(f"✅ Loaded API key from: {default_env}")
                return os.environ["OPENAI_API_KEY"]
        
        # Try current directory
        if Path(".env").exists():
            load_dotenv(".env")
            if os.environ.get("OPENAI_API_KEY"):
                print("✅ Loaded API key from: ./.env")
                return os.environ["OPENAI_API_KEY"]
                
    except ImportError:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    
    # Last resort: check system environment
    if os.environ.get("OPENAI_API_KEY"):
        print("✅ Using API key from system environment")
        return os.environ["OPENAI_API_KEY"]
    
    # Not found anywhere
    raise ValueError(
        "OpenAI API key not found!\n"
        "Please set it in one of these ways:\n"
        "1. Create a .env file with: OPENAI_API_KEY=your-key-here\n"
        "2. Set environment variable: export OPENAI_API_KEY=your-key-here\n"
        "3. Pass it to load_api_key(env_file_path='/path/to/.env')"
    )


def setup_environment(env_file_path: str = None) -> None:
    """
    Setup environment variables for all services.
    
    Args:
        env_file_path: Optional path to .env file
    """
    try:
        api_key = load_api_key(env_file_path)
        os.environ["OPENAI_API_KEY"] = api_key
        print("🔐 Environment configured successfully")
    except ValueError as e:
        print(f"❌ Error: {e}")
        raise
