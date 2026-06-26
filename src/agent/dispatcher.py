import os
import json

try:
    from agno.agent import Agent # type: ignore
    from agno.models.openai import OpenAIChat # type: ignore
    HAS_AGNO = True
except ImportError:
    HAS_AGNO = False

from src.agent.tools import calculate_face_quality, detect_adversarial_noise, extract_metadata, detect_text_density # type: ignore

class DispatcherAgent:
    """
    Intelligent Switchboard using Agno (Phidata).
    Triage inputs based on Modality, Adversarial Noise, and Image Quality.
    """
    def __init__(self, api_key=None, model_id="gpt-3.5-turbo"):
        # For a truly local fallback, you could use Ollama here instead:
        # from agno.models.ollama import Ollama
        # model = Ollama(id="llama3")
        
        # If no API key is provided, we use a structured LLM router
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # We define an Agno Agent equipped with our deterministic computer vision tools
        if HAS_AGNO and self.api_key:
            self.agent = Agent(
                model=OpenAIChat(id=model_id, api_key=self.api_key),
                tools=[calculate_face_quality, detect_adversarial_noise, extract_metadata, detect_text_density],
                description="You are the Intelligent Switchboard Dispatcher for the DeepScan Deepfake Detection Pipeline.",
                instructions=[
                    "Analyze the provided file path using your tools.",
                    "1. First extract metadata to find file extension and modality.",
                    "2. If modality is 'text', route immediately to TEXT_TAMPER.",
                    "3. If it is an image, check the text density using detect_text_density. If is_document is True, route to TEXT_TAMPER.",
                    "4. If it is a standard image, strictly check for Adversarial Noise.",
                    "5. If it is a standard image, check the Face Quality.",
                    "6. Return a JSON response structured EXACTLY like this: " +
                    "{\"route\": \"TEXT_TAMPER\" | \"FACE_PIPELINE\" | \"REJECT\" | \"VIDEO_PIPELINE\", \"confidence\": <float>, \"reason\": \"<your reasoning>\", \"status_code\": <string>}",
                    "If Face Quality 'is_sharp' is False, route = REJECT with reason 'Too Blurry'.",
                    "If Adversarial 'attack_suspected' is True, append FLAG to reason but still route to FACE_PIPELINE."
                ],
                markdown=False
            )
        else:
            self.agent = None
        
    def dispatch(self, file_path: str) -> dict:
        """
        Takes a file, runs it through the Agent's reasoning, and returns the target route.
        """
        # --- Fallback/Mock mode for Local Only execution if no API KEY or Agno is present ---
        if not self.agent or not self.api_key:
            return self._mock_dispatch_deterministic(file_path)
            
        try:
            prompt = f"Please determine the pipeline route for this file: {file_path}"
            # Run the agent synchronously
            response = self.agent.run(prompt)
            
            # The agent should return a JSON string based on the instructions
            # Let's extract and parse it
            content = response.content
            # Very basic JSON extraction from string if it contains markdown ticks
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                return json.loads(json_str)
            else:
                return {
                    "route": "FACE_PIPELINE", # Safe default
                    "confidence": 0.5,
                    "reason": "Failed to parse agent JSON. Defaulting to face pipeline.",
                    "raw_response": content
                }
        except Exception as e:
            print(f"Agent Router failed: {e}")
            return self._mock_dispatch_deterministic(file_path)
            
    def _mock_dispatch_deterministic(self, file_path: str) -> dict:
        """
        A deterministic heuristic fallback if the LLM API is unavailable.
        It acts like a fast rule-based agent using the same tools.
        """
        print("[Dispatcher] Running local heuristic tools (No API key or Agno found)")
        meta = extract_metadata(file_path)
        
        if meta.get("error"):
            return {"route": "REJECT", "reason": "File not found", "confidence": 1.0}
            
        if meta["modality"] == "video":
            return {"route": "VIDEO_PIPELINE", "reason": "Standard video file detected", "confidence": 0.9}
            
        if meta["modality"] == "text":
            return {"route": "TEXT_TAMPER", "reason": f"Text file input detected ({meta.get('extension')})", "confidence": 0.95}
            
        # It's an image. Use Scout Tools.
        # Check text density first to see if it's a document/text-heavy image
        text_info = detect_text_density(file_path)
        if not text_info.get("error") and text_info.get("is_document", False):
            return {
                "route": "TEXT_TAMPER",
                "reason": f"Text-heavy document image detected (Density: {text_info.get('text_density', 0):.2%})",
                "confidence": 0.9
            }
            
        quality = calculate_face_quality(file_path)
        noise = detect_adversarial_noise(file_path)
        
        reason = ""
        route = "FACE_PIPELINE" # Default for image
        
        if not quality.get("is_sharp", True):
            return {
                "route": "REJECT", 
                "reason": f"Image is too blurry (Variance: {quality.get('variance_score', 0):.2f}). Dropping to save compute.", 
                "confidence": 0.99
            }
            
        if noise.get("attack_suspected", False):
            reason += f"FLAG: High pixel-noise detected (Level: {noise.get('noise_level', 0):.2f}). Potential adversarial attack. "
        else:
            reason += "Image looks completely clean. "
            
        return {
            "route": route,
            "reason": reason.strip(),
            "confidence": 0.85
        }
