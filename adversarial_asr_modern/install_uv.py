#!/usr/bin/env python3
"""
UV installer and setup script for adversarial ASR modern.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def install_uv():
    """Install UV package manager."""
    print("Installing UV package manager...")
    
    system = platform.system().lower()
    
    try:
        if system in ['linux', 'darwin']:  # Linux or macOS
            # Use the official installer
            cmd = ['curl', '-LsSf', 'https://astral.sh/uv/install.sh']
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, _ = process.communicate()
            
            if process.returncode == 0:
                # Pipe to shell for installation
                install_process = subprocess.run(['sh'], input=output, check=True)
                print("✅ UV installed successfully!")
                return True
            else:
                print("❌ Failed to download UV installer")
                return False
                
        elif system == 'windows':
            # Windows installation
            cmd = [
                'powershell', '-c', 
                'irm https://astral.sh/uv/install.sh | iex'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ UV installed successfully!")
                return True
            else:
                print(f"❌ Failed to install UV on Windows: {result.stderr}")
                return False
        else:
            print(f"❌ Unsupported system: {system}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing UV: {e}")
        return False

def check_uv_installed():
    """Check if UV is already installed."""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ UV already installed: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def add_uv_to_path():
    """Add UV to PATH if needed."""
    home = Path.home()
    uv_bin = home / ".local" / "bin"
    
    if uv_bin.exists():
        current_path = os.environ.get('PATH', '')
        if str(uv_bin) not in current_path:
            print(f"Adding {uv_bin} to PATH...")
            os.environ['PATH'] = f"{uv_bin}:{current_path}"
            
            # Suggest permanent addition
            shell = os.environ.get('SHELL', '').split('/')[-1]
            if shell in ['bash', 'zsh']:
                config_file = home / f".{shell}rc"
                print(f"\n💡 To make UV permanently available, add this to {config_file}:")
                print(f'export PATH="{uv_bin}:$PATH"')
            
            return True
    return False

def setup_project():
    """Set up the project with UV."""
    print("\nSetting up project with UV...")
    
    try:
        # Sync dependencies
        print("Running 'uv sync'...")
        result = subprocess.run(['uv', 'sync'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to sync dependencies:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error setting up project: {e}")
        return False

def test_installation():
    """Test the UV installation with our project."""
    print("\nTesting installation...")
    
    try:
        # Test UV python
        result = subprocess.run(['uv', 'run', 'python', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ UV Python: {result.stdout.strip()}")
        else:
            print(f"❌ UV Python test failed: {result.stderr}")
            return False
        
        # Test our installation script
        result = subprocess.run(['uv', 'run', 'python', 'test_installation.py'], 
                              capture_output=True, text=True)
        
        print("Installation test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error testing installation: {e}")
        return False

def main():
    """Main installation workflow."""
    print("=" * 60)
    print("UV INSTALLATION AND SETUP")
    print("=" * 60)
    
    # Check if UV is already installed
    if check_uv_installed():
        print("UV is already installed!")
    else:
        print("UV not found. Installing...")
        if not install_uv():
            print("\n❌ UV installation failed.")
            print("\nManual installation instructions:")
            print("macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
            print("Windows: powershell -c \"irm https://astral.sh/uv/install.sh | iex\"")
            return
        
        # Add to PATH if needed
        add_uv_to_path()
        
        # Check again after installation
        if not check_uv_installed():
            print("\n⚠️  UV installed but not found in PATH.")
            print("Please restart your terminal and try again.")
            return
    
    # Set up the project
    if setup_project():
        print("\n🎉 Project setup complete!")
        
        # Test the installation
        if test_installation():
            print("\n✅ Everything working! You can now run:")
            print("  uv run python run_ctc_attack.py --test_model")
            print("  uv run python run_ctc_attack.py --num_examples 3")
        else:
            print("\n⚠️  Setup complete but tests failed. Check the output above.")
    else:
        print("\n❌ Project setup failed. See errors above.")

if __name__ == "__main__":
    main()
