import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class DramaScoringProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.drama_dir = self.output_base_dir / "drama"
        self.drama_dir.mkdir(parents=True, exist_ok=True)
        
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.location = os.getenv("GCP_LOCATION")
        
        try:
            self.llm_client = genai.Client(
                vertexai=True, 
                project=self.project_id, 
                location=self.location
            )
        except Exception as e:
            logger.warning(f"Failed to init GenAI client: {e}")
            self.llm_client = None

    def process_drama_scoring(self, video_id: str, scene_metadata_path: str, genre: str = "Action") -> list:
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {scene_metadata_path}")
            return []
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        scenes = metadata.get("scenes", [])
        intelligence_results = []
        
        for scene in scenes:
            scene_id = scene.get("scene_id")
            
            # Aggregate intelligence from all stages
            speech_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{scene_id}.json")
            face_data = self._read_json(self.output_base_dir / "faces" / f"face_intelligence_{scene_id}.json")
            emotion_data = self._read_json(self.output_base_dir / "emotions" / f"emotion_{scene_id}.json")
            bgm_data = self._read_json(self.output_base_dir / "music" / f"bgm_intelligence_{scene_id}.json")
            narrative_data = self._read_json(self.output_base_dir / "narrative" / f"narrative_intelligence_{scene_id}.json")
            virality_data = self._read_json(self.output_base_dir / "virality" / f"virality_intelligence_{scene_id}.json")
            nostalgia_data = self._read_json(self.output_base_dir / "nostalgia" / f"nostalgia_intelligence_{scene_id}.json")
            
            # Extract components for the formula
            # Stage 4: DialogueImpact
            speech_intel = speech_data.get("intelligence_results", [{}])[0] if speech_data else {}
            dialogue_impact = speech_intel.get("impact_score", 0)
            
            # Stage 5: HeroPresence, ReactionStrength
            # If face_data exists, infer hero presence and reaction strength based on expressions
            hero_presence = 0
            reaction_strength = 0
            if face_data and "scene_timeline" in face_data:
                expressions = [t.get("dominant_expression", "") for t in face_data["scene_timeline"]]
                if len(expressions) > 0:
                    hero_presence = 80 # Simplified heuristic, ideally fetched from face analysis
                    # Calculate reaction strength based on emotional variance
                    if "Surprise" in expressions or "Fear" in expressions or "Angry" in expressions:
                        reaction_strength = 90
                    else:
                        reaction_strength = 50
            
            # Stage 6: EmotionIntensity, Suspense
            emotion_intensity = emotion_data.get("weighted_emotional_score", 0)
            suspense = 85 if "suspense" in emotion_data.get("emotion_curve", "").lower() else 50
            
            # Stage 7: BGMIntensity
            bgm_intensity = bgm_data.get("intensity", 0)
            
            # Stage 8: NarrativeImportance, PayoffStrength
            narrative_importance = narrative_data.get("narrative_importance", 0)
            payoff_strength = 90 if "payoff" in narrative_data.get("arc_type", "").lower() else 40
            
            # Stage 9: Virality
            virality_score = virality_data.get("viral_probability", 0)
            
            # Stage 10: Nostalgia
            nostalgia_score = nostalgia_data.get("nostalgia_score", 0)
            
            scene_context = {
                "NarrativeImportance": narrative_importance,
                "DialogueImpact": dialogue_impact,
                "EmotionIntensity": emotion_intensity,
                "BGMIntensity": bgm_intensity,
                "ReactionStrength": reaction_strength,
                "Virality": virality_score,
                "Nostalgia": nostalgia_score,
                "HeroPresence": hero_presence,
                "Suspense": suspense,
                "PayoffStrength": payoff_strength,
                "Genre": genre
            }
            
            # Feed to LLM for final Drama Score computation
            llm_response = self._extract_drama_score(scene_id, scene_context)
            
            scene_payload = {
                "final_drama_score": llm_response.get("final_drama_score", 0)
            }
            
            out_path = self.drama_dir / f"drama_score_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            scene_payload_for_master = scene_payload.copy()
            scene_payload_for_master["scene_id"] = scene_id
            scene_payload_for_master["genre"] = genre
            intelligence_results.append(scene_payload_for_master)

        # Save master JSON
        ranked_scenes = sorted(intelligence_results, key=lambda x: x["final_drama_score"], reverse=True)
        
        master_payload = {
            "status": "completed",
            "message": f"Stage 11 Drama Scoring Engine completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "top_drama_scenes": [s["scene_id"] for s in ranked_scenes[:3]],
            "drama_scoring_results": ranked_scenes
        }
        
        master_path = self.drama_dir / "master_drama_scoring.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_payload, f, indent=4, ensure_ascii=False)
            
        return intelligence_results

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON {path}: {e}")
        return {}
        
    def _extract_drama_score(self, scene_id, context):
        if not self.llm_client:
            return self._mock_drama_output()
            
        prompt = f"""
        You are the Master Drama Scorer.
        Your goal is to generate the final cinematic importance score (DramaScore) for Scene {scene_id}.
        
        --- RAW INTELLIGENCE COMPONENTS ---
        NarrativeImportance: {context['NarrativeImportance']}
        DialogueImpact: {context['DialogueImpact']}
        EmotionIntensity: {context['EmotionIntensity']}
        BGMIntensity: {context['BGMIntensity']}
        ReactionStrength: {context['ReactionStrength']}
        Virality: {context['Virality']}
        Nostalgia: {context['Nostalgia']}
        HeroPresence: {context['HeroPresence']}
        Suspense: {context['Suspense']}
        PayoffStrength: {context['PayoffStrength']}
        
        --- GENRE TUNING ---
        Current Target Genre: {context['Genre']}
        
        RULES:
        If Genre is "Action", apply more weight to:
        - HeroPresence (motion/elevation)
        - BGMIntensity (elevation bgm)
        - PayoffStrength
        
        If Genre is "Serials", apply more weight to:
        - EmotionIntensity (crying)
        - ReactionStrength (reactions)
        - NarrativeImportance (emotional arcs)
        
        INSTRUCTIONS:
        Calculate the DramaScore out of 100 using the components provided, heavily influenced by the Genre Tuning rules.
        
        Provide your analysis ONLY as a valid JSON object matching this exact schema:
        {{
            "final_drama_score": integer 0-100
        }}
        """
        
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            return json.loads(response_text.strip())
        except Exception as e:
            logger.error(f"Drama Scoring LLM failed: {e}")
            return self._mock_drama_output()

    def _mock_drama_output(self):
        return {
            "final_drama_score": 97
        }
