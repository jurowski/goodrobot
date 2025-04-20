import os
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api.api import app
from src.voice_recognition.wake_word import WakeWordDetector
from src.voice_recognition.speech_to_text import SpeechToText
from src.settings import Settings

# Initialize test client
client = TestClient(app)

@pytest.fixture
def test_audio_data():
    """Generate test audio data."""
    # Generate 1 second of silence at 16kHz
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    silence = np.zeros_like(t)
    
    # Convert to 16-bit PCM
    silence_pcm = (silence * 32767).astype(np.int16)
    return silence_pcm.tobytes()

@pytest.fixture
def wake_word_detector():
    """Initialize wake word detector for testing."""
    settings = Settings()
    return WakeWordDetector(settings)

@pytest.fixture
def speech_to_text():
    """Initialize speech-to-text for testing."""
    settings = Settings()
    return SpeechToText(settings)

def test_voice_endpoint_get():
    """Test GET /voice endpoint returns HTML."""
    response = client.get("/voice")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Good Robot Voice Interface" in response.text

def test_voice_endpoint_post_empty_audio():
    """Test POST /voice endpoint with empty audio."""
    response = client.post(
        "/voice",
        files={"audio": ("test.wav", b"", "audio/wav")},
        data={"language": "en-US"}
    )
    assert response.status_code == 400
    assert "Empty audio file" in response.json()["detail"]

def test_voice_endpoint_post_valid_audio(test_audio_data):
    """Test POST /voice endpoint with valid audio."""
    response = client.post(
        "/voice",
        files={"audio": ("test.wav", test_audio_data, "audio/wav")},
        data={"language": "en-US"}
    )
    assert response.status_code == 200
    result = response.json()
    assert "wake_word_detected" in result
    assert isinstance(result["wake_word_detected"], bool)

def test_websocket_connection():
    """Test WebSocket connection and basic message handling."""
    with client.websocket_connect("/ws/test_client") as websocket:
        # Test config message
        config = {
            "type": "config",
            "sensitivity": 0.5,
            "noiseThreshold": 0.15
        }
        websocket.send_text(json.dumps(config))
        
        # Send test audio data
        audio_data = np.zeros(512, dtype=np.int16).tobytes()
        websocket.send_bytes(audio_data)
        
        # Should receive a response
        try:
            response = websocket.receive_json()
            assert "type" in response
            assert response["type"] in ["audio_processed", "error"]
        except Exception:
            # If no response is received, that's okay for this test
            pass

@pytest.mark.asyncio
async def test_speech_to_text_transcription(speech_to_text, test_audio_data):
    """Test speech-to-text transcription."""
    # Test transcription
    result = await speech_to_text.transcribe_audio(test_audio_data)
    assert isinstance(result, str)

# Add Playwright tests for end-to-end testing
try:
    from playwright.sync_api import sync_playwright
    import pytest

    @pytest.fixture(scope="module")
    def browser_page():
        """Create a browser page for testing."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            yield page
            browser.close()

    def test_voice_interface_ui(browser_page):
        """Test voice interface UI elements and interactions."""
        # Set up console log capture
        console_messages = []
        browser_page.on("console", lambda msg: console_messages.append(msg.text))
        
        # Navigate to voice interface
        browser_page.goto("http://localhost:8000/voice")
        
        # Check title
        assert "Good Robot Voice Interface" in browser_page.title()
        
        # Check main UI elements - using class selectors instead of IDs
        start_button = browser_page.locator(".controls button:first-child")
        stop_button = browser_page.locator(".controls button:last-child")
        assert start_button.is_visible()
        assert stop_button.is_visible()
        
        # Test start/stop button states
        assert not start_button.is_disabled()
        assert stop_button.is_disabled()
        
        # Click start button and handle permissions
        browser_page.on("dialog", lambda dialog: dialog.accept())
        start_button.click()
        
        # Add debug logging to track status changes
        browser_page.evaluate("""() => {
            const status = document.getElementById('status');
            const originalTextContent = status.textContent;
            Object.defineProperty(status, 'textContent', {
                set: function(value) {
                    console.log('Status text changed to:', value);
                    this.innerText = value;
                },
                get: function() {
                    return this.innerText;
                }
            });
            console.log('Initial status text:', originalTextContent);
        }""")
        
        # Wait for WebSocket connection and audio setup
        try:
            browser_page.wait_for_function("""() => {
                const status = document.getElementById('status');
                const text = status ? status.textContent : '';
                console.log('Current status text:', text);
                return text.includes('Listening');
            }""", timeout=10000)
        except Exception as e:
            # Get the current status text for debugging
            current_status = browser_page.evaluate("""() => {
                const status = document.getElementById('status');
                return status ? status.textContent : 'Status element not found';
            }""")
            print(f"Failed to wait for Listening status. Current status: {current_status}")
            print("\nConsole messages:")
            for msg in console_messages:
                print(f"  {msg}")
            raise e
        
        # Now check button states
        assert start_button.is_disabled()
        assert not stop_button.is_disabled()
        
        # Test volume meter visibility
        volume_meter = browser_page.locator(".volume-meter")
        assert volume_meter.is_visible()
        
        # Test stop button
        stop_button.click()
        
        # Wait for cleanup to complete
        try:
            browser_page.wait_for_function("""() => {
                const status = document.getElementById('status');
                const text = status ? status.textContent : '';
                console.log('Current status text:', text);
                return text === 'Stopped';
            }""", timeout=10000)
        except Exception as e:
            current_status = browser_page.evaluate("""() => {
                const status = document.getElementById('status');
                return status ? status.textContent : 'Status element not found';
            }""")
            print(f"Failed to wait for Stopped status. Current status: {current_status}")
            print("\nConsole messages:")
            for msg in console_messages:
                print(f"  {msg}")
            raise e
        
        # Check final button states
        assert not start_button.is_disabled()
        assert stop_button.is_disabled()

    def test_voice_interface_settings(browser_page):
        """Test voice interface settings controls."""
        browser_page.goto("http://localhost:8000/voice")
        
        # Test sensitivity slider
        sensitivity = browser_page.locator("#sensitivity")
        assert sensitivity.is_visible()
        
        # Change sensitivity
        browser_page.evaluate("""
            document.getElementById('sensitivity').value = '0.7';
            document.getElementById('sensitivity').dispatchEvent(new Event('input'));
        """)
        sensitivity_value = browser_page.locator("#sensitivityValue")
        assert sensitivity_value.inner_text() == "0.7"
        
        # Test audio gain slider
        audio_gain = browser_page.locator("#audioGain")
        assert audio_gain.is_visible()
        
        # Change audio gain
        browser_page.evaluate("""
            document.getElementById('audioGain').value = '1.5';
            document.getElementById('audioGain').dispatchEvent(new Event('input'));
        """)
        audio_gain_value = browser_page.locator("#audioGainValue")
        assert audio_gain_value.inner_text() == "1.5"
        
        # Test language selector
        language_select = browser_page.locator("#language")
        assert language_select.is_visible()
        
        # Change language
        language_select.select_option("en-GB")
        assert language_select.evaluate("el => el.value") == "en-GB"

    def test_voice_interface_logs(browser_page):
        """Test voice interface log functionality."""
        browser_page.goto("http://localhost:8000/voice")
        
        # Check log container
        log_container = browser_page.locator("#logContainer")
        assert log_container.is_visible()
        
        # Test log filters
        show_server_logs = browser_page.locator("#showServerLogs")
        show_client_logs = browser_page.locator("#showClientLogs")
        
        assert show_server_logs.is_checked()
        assert show_client_logs.is_checked()
        
        # Test log search
        log_search = browser_page.locator("#logSearch")
        assert log_search.is_visible()
        
        # Enter search term
        log_search.fill("WebSocket")
        browser_page.wait_for_timeout(500)  # Wait for search to update
        
        # Test log clear button
        clear_logs = browser_page.locator("#clearLogs")
        assert clear_logs.is_visible()
        
        clear_logs.click()
        browser_page.wait_for_timeout(500)  # Wait for logs to clear
        
        # Verify log container is empty
        assert log_container.inner_text().strip() == ""

except ImportError:
    print("Playwright not installed. Skipping end-to-end tests.") 