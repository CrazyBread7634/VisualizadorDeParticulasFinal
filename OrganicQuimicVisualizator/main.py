# Archivo: main.py - CON INTEGRACIÓN GEMINI
# Sistema de visualización molecular con RDKit controlado por gestos + Gemini AI

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import os
import mediapipe as mp
import math
import numpy as np
import base64
import io
import time
import json
import re  
import requests
from PIL import Image, ImageDraw, ImageFont

# Importaciones de RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdMolAlign import AlignMol
    from rdkit.Chem import rdFMCS
    RDKIT_AVAILABLE = True
    print("✅ RDKit importado correctamente")
except ImportError as e:
    print(f"⚠️ RDKit no está disponible: {e}")
    print("Instalalo con: pip install rdkit")
    RDKIT_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuración de directorios
UPLOAD_FOLDER = 'static/uploads'
MOLECULES_FOLDER = 'static/molecules'
for folder in [UPLOAD_FOLDER, MOLECULES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ===== CONFIGURACIÓN DE GEMINI AI =====
GEMINI_API_KEY = "AIzaSyAox4uJYUeLZet5YrR7R7BT7q0vmiluI4w"  # API Key de tu compañero
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Modelos disponibles (basados en el código de tu compañero)
GEMINI_MODELS = {
    'gemini-1.5-flash': 'gemini-1.5-flash-latest',
    'gemini-1.5-pro': 'gemini-1.5-pro-latest', 
    'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite-latest',
    'gemini-2.5-pro': 'gemini-2.5-pro-latest'
}

class GeminiMolecularAnalyzer:
    """Clase para análisis molecular usando Gemini AI - Adaptada del proyecto del compañero"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = GEMINI_BASE_URL
        
    def analyze_molecule(self, molecule_data, model='gemini-1.5-flash'):
        """Genera análisis químico completo de una molécula usando Gemini"""
        try:
            if not molecule_data or not molecule_data.get('propiedades'):
                return {
                    'error': True,
                    'message': "No hay datos moleculares disponibles para analizar."
                }
            
            props = molecule_data['propiedades']
            smiles = molecule_data.get('smiles', 'No disponible')
            
            # Prompt especializado basado en el estilo del proyecto del compañero
            prompt = f"""Eres un químico medicinal y farmacólogo molecular experto. Analiza la molécula con SMILES "{smiles}" y proporciona un informe detallado profesional.

DATOS MOLECULARES:
- Nombre: {props.get('nombre', 'Desconocido')}
- Fórmula: {props.get('formula', 'N/A')}
- Peso Molecular: {props.get('peso_molecular', 'N/A')} Da
- LogP: {props.get('logp', 'N/A')}
- TPSA: {props.get('tpsa', 'N/A')} Ų
- Donadores H: {props.get('hbd', 'N/A')}
- Aceptores H: {props.get('hba', 'N/A')}
- Enlaces Rotables: {props.get('rotatable_bonds', 'N/A')}
- Anillos Aromáticos: {props.get('aromatic_rings', 'N/A')}

Genera una respuesta en formato JSON con la siguiente estructura exacta:
{{
  "suggestedName": "Un nombre químico apropiado y científicamente razonable.",
  "keyChemicalFeatures": "Resumen técnico de características estructurales: grupos funcionales, puntos de interacción, sitios reactivos.",
  "potentialPharmacologicalProperties": "Explicación de posibles propiedades biológicas y farmacológicas, considerando su perfil sobre receptores relevantes.",
  "potentialUses": "Posibles usos farmacológicos o áreas de investigación relevantes.",
  "lipinskiAnalysis": "Análisis de regla de Lipinski y drug-likeness.",
  "structuralInsights": "Observaciones sobre la estructura 3D y conformación molecular."
}}

No incluyas explicaciones adicionales fuera del formato JSON. Redacta cada campo como parte de un informe científico."""

            # Llamar a Gemini usando el método del proyecto del compañero
            model_name = GEMINI_MODELS.get(model, 'gemini-1.5-flash-latest')
            response = self._call_gemini_api(prompt, model_name)
            
            if response:
                try:
                    # Limpiar respuesta (igual que en combination.js)
                    json_string = re.sub(r"```(json)?", "", response).strip()
                    analysis = json.loads(json_string)
                    
                    return {
                        'error': False,
                        'analysis': analysis
                    }
                except json.JSONDecodeError as e:
                    print(f"❌ Error parseando JSON de Gemini: {e}")
                    return {
                        'error': True,
                        'message': f"Error procesando respuesta de IA: {str(e)}"
                    }
            else:
                return {
                    'error': True,
                    'message': "Error al generar análisis con Gemini AI."
                }
                
        except Exception as e:
            print(f"❌ Error en análisis Gemini: {e}")
            return {
                'error': True,
                'message': f"Error inesperado: {str(e)}"
            }
    
    def analyze_molecular_interaction(self, ligando_data, receptor_data, docking_score, model='gemini-1.5-flash'):
        """Analiza interacción entre ligando y receptor"""
        try:
            if not ligando_data or not receptor_data:
                return {
                    'error': True,
                    'message': "Necesita cargar tanto ligando como receptor para el análisis."
                }
            
            ligando_props = ligando_data.get('propiedades', {})
            receptor_props = receptor_data.get('propiedades', {})
            ligando_smiles = ligando_data.get('smiles', 'N/A')
            receptor_smiles = receptor_data.get('smiles', 'N/A')
            
            # Prompt para análisis de interacción (estilo del compañero)
            prompt = f"""Eres un experto en química orgánica y diseño racional de fármacos. Analiza la interacción entre estas dos moléculas.

LIGANDO: {ligando_smiles}
- Nombre: {ligando_props.get('nombre', 'Desconocido')}
- MW: {ligando_props.get('peso_molecular', 'N/A')} Da
- LogP: {ligando_props.get('logp', 'N/A')}
- HBD/HBA: {ligando_props.get('hbd', 'N/A')}/{ligando_props.get('hba', 'N/A')}

RECEPTOR: {receptor_smiles}
- Nombre: {receptor_props.get('nombre', 'Desconocido')}  
- MW: {receptor_props.get('peso_molecular', 'N/A')} Da
- LogP: {receptor_props.get('logp', 'N/A')}
- HBD/HBA: {receptor_props.get('hbd', 'N/A')}/{receptor_props.get('hba', 'N/A')}

SCORE DE DOCKING: {docking_score:.1f}

Genera una respuesta en formato JSON con la siguiente estructura:
{{
  "interactionAnalysis": "Análisis de complementariedad y compatibilidad entre las moléculas.",
  "probableInteractions": "Tipos de unión química probables (H-bonds, hidrofóbicas, van der Waals).",
  "scoreInterpretation": "Interpretación del score de afinidad molecular.",
  "optimizationSuggestions": "Sugerencias para mejorar la interacción.",
  "biologicalRelevance": "Relevancia biológica y farmacológica de esta interacción."
}}

Responde solo en formato JSON, sin texto adicional."""

            model_name = GEMINI_MODELS.get(model, 'gemini-1.5-flash-latest')
            response = self._call_gemini_api(prompt, model_name)
            
            if response:
                try:
                    json_string = re.sub(r"```(json)?", "", response).strip()
                    analysis = json.loads(json_string)
                    
                    return {
                        'error': False,
                        'interaction_analysis': analysis
                    }
                except json.JSONDecodeError as e:
                    return {
                        'error': True,
                        'message': f"Error procesando análisis de interacción: {str(e)}"
                    }
            else:
                return {
                    'error': True,
                    'message': "Error generando análisis de interacción."
                }
                
        except Exception as e:
            print(f"❌ Error en análisis de interacción: {e}")
            return {
                'error': True,
                'message': f"Error: {str(e)}"
            }
    
    def generate_gesture_suggestions(self, gesture_mode, molecule_data, model='gemini-2.5-flash-lite'):
        """Genera sugerencias contextuales basadas en gestos actuales"""
        try:
            mol_name = molecule_data.get('propiedades', {}).get('nombre', 'Desconocida') if molecule_data else 'Ninguna'
            mol_smiles = molecule_data.get('smiles', 'N/A') if molecule_data else 'N/A'
            
            prompt = f"""Como experto en interfaces gestuales para química computacional, proporciona sugerencias útiles.

CONTEXTO ACTUAL:
- Modo de gesto: {gesture_mode}
- Molécula visualizada: {mol_name}
- SMILES: {mol_smiles}

Genera una respuesta en formato JSON:
{{
  "gestureActions": "Qué acciones puede realizar con sus gestos actuales (3-4 puntos específicos).",
  "chemicalObservations": "Qué aspectos químicos específicos observar en la visualización actual.",
  "interactionTips": "Consejos para manipular la vista 3D eficientemente.",
  "nextSteps": "Sugerencias de próximos pasos en el análisis molecular."
}}

Responde en español, formato JSON puro, sugerencias prácticas y específicas."""

            # Usar modelo rápido para sugerencias (como en el proyecto del compañero)
            model_name = GEMINI_MODELS.get(model, 'gemini-2.5-flash-lite-latest')
            response = self._call_gemini_api(prompt, model_name)
            
            if response:
                try:
                    json_string = re.sub(r"```(json)?", "", response).strip()
                    suggestions = json.loads(json_string)
                    
                    return {
                        'error': False,
                        'suggestions': suggestions
                    }
                except json.JSONDecodeError:
                    return {
                        'error': False,
                        'suggestions': {
                            'gestureActions': f"Modo {gesture_mode}: Continúa explorando la molécula con tus gestos.",
                            'chemicalObservations': "Observa la distribución de átomos y enlaces en el espacio 3D.",
                            'interactionTips': "Usa gestos suaves para mejor control de la visualización.",
                            'nextSteps': "Considera cargar una segunda molécula para análisis comparativo."
                        }
                    }
            else:
                return {
                    'error': False,
                    'suggestions': {
                        'gestureActions': "Usa tus manos para controlar la visualización molecular.",
                        'chemicalObservations': "Explora la estructura 3D de la molécula.",
                        'interactionTips': "Experimenta con diferentes modos de visualización.",
                        'nextSteps': "Carga diferentes moléculas para comparar estructuras."
                    }
                }
                
        except Exception as e:
            print(f"❌ Error en sugerencias gestuales: {e}")
            return {
                'error': False,
                'suggestions': {
                    'gestureActions': "Controla la vista molecular con gestos de mano.",
                    'chemicalObservations': "Analiza la estructura molecular en 3D.",
                    'interactionTips': "Usa movimientos suaves para mejor control.",
                    'nextSteps': "Explora diferentes moléculas y sus propiedades."
                }
            }
    
    def _call_gemini_api(self, prompt, model_name):
        """Realiza llamada a la API de Gemini - Método adaptado del proyecto del compañero"""
        try:
            url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 1000,
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    print("❌ Respuesta vacía de Gemini")
                    return None
            else:
                print(f"❌ Error API Gemini: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Timeout en llamada a Gemini")
            return None
        except Exception as e:
            print(f"❌ Error llamando a Gemini: {e}")
            return None

# Instancia global del analizador Gemini
gemini_analyzer = GeminiMolecularAnalyzer(GEMINI_API_KEY)

# ===== RESTO DEL CÓDIGO ORIGINAL (MediaPipe, MolecularViewer, etc.) =====
# Mantener todo el código original de MediaPipe y MolecularViewer...

mp_hands = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def detectar_gesto_agarre(landmarks, w, h):
    """Detecta gesto de agarre"""
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    dedos_doblados = 0
    
    for i in range(1, 5):
        tip = landmarks[tip_ids[i]]
        pip = landmarks[pip_ids[i]]
        if tip.y > pip.y:
            dedos_doblados += 1
    
    thumb_tip = landmarks[tip_ids[0]]
    thumb_pip = landmarks[pip_ids[0]]
    if thumb_tip.x > thumb_pip.x:
        dedos_doblados += 1
    
    return dedos_doblados >= 3

# ===== CLASE MOLECULAR VIEWER ORIGINAL + EXTENSIONES GEMINI =====
class MolecularViewer:
    def __init__(self):
        self.mol_ligando = None
        self.mol_receptor = None
        self.conformaciones = {}
        self.propiedades = {}
        
        # Estados de visualización
        self.viewer_state = {
            'rotacion': {'x': 0, 'y': 0, 'z': 0},
            'traslacion': {'x': 0, 'y': 0, 'z': 0},
            'zoom': 1.0,
            'modo_vista': '3d',  
            'colores': 'cpk',
            'mostrar_hidrogenos': False,
            'estilo_enlace': 'stick'
        }
        
        # Control por gestos
        self.gesto_state = {
            'modo_gesto': 'libre',
            'manos_detectadas': 0,
            'ligando_activo': False,
            'receptor_activo': False
        }
        
        # ===== NUEVO: Estado de AI Gemini =====
        self.ai_state = {
            'last_analysis': None,
            'last_interaction_analysis': None,
            'current_suggestions': None,
            'selected_model': 'gemini-1.5-flash',
            'analysis_available': False
        }
    
    # Mantener todos los métodos originales...
    def cargar_molecula_desde_smiles(self, smiles, nombre="Molecula"):
        """Carga una molécula desde SMILES - MÉTODO ORIGINAL"""
        if not RDKIT_AVAILABLE:
            return None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"❌ No se pudo parsear SMILES: {smiles}")
                return None
                
            mol = Chem.AddHs(mol)
            
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.UFFOptimizeMolecule(mol)
            except Exception as e:
                print(f"⚠️ Error en optimización 3D, usando 2D: {e}")
                AllChem.Compute2DCoords(mol)
            
            propiedades = self.calcular_propiedades_seguras(mol, nombre)
            
            return {
                'mol': mol,
                'propiedades': propiedades,
                'smiles': smiles
            }
            
        except Exception as e:
            print(f"❌ Error cargando molécula: {e}")
            return None
    
    def calcular_propiedades_seguras(self, mol, nombre="Molecula"):
        """Calcula propiedades moleculares con manejo de errores - MÉTODO ORIGINAL"""
        try:
            propiedades = {
                'nombre': nombre,
                'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                'peso_molecular': round(Descriptors.MolWt(mol), 2),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds()
            }
            
            try:
                propiedades['logp'] = round(Descriptors.MolLogP(mol), 2)
            except:
                propiedades['logp'] = 0.0
                
            try:
                propiedades['tpsa'] = round(Descriptors.TPSA(mol), 2)
            except:
                propiedades['tpsa'] = 0.0
                
            try:
                propiedades['hbd'] = Descriptors.NumHDonors(mol)
                propiedades['hba'] = Descriptors.NumHAcceptors(mol)
            except:
                propiedades['hbd'] = 0
                propiedades['hba'] = 0
                
            try:
                propiedades['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            except:
                propiedades['rotatable_bonds'] = 0
                
            try:
                propiedades['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            except:
                propiedades['aromatic_rings'] = 0
            
            return propiedades
            
        except Exception as e:
            print(f"⚠️ Error calculando propiedades: {e}")
            return {
                'nombre': nombre,
                'formula': 'Unknown',
                'peso_molecular': 0.0,
                'logp': 0.0,
                'tpsa': 0.0,
                'hbd': 0,
                'hba': 0,
                'rotatable_bonds': 0,
                'aromatic_rings': 0,
                'num_atoms': mol.GetNumAtoms() if mol else 0,
                'num_bonds': mol.GetNumBonds() if mol else 0
            }
    
    # ===== AGREGAR ESTOS MÉTODOS A LA CLASE MolecularViewer =====
# Insertar después del método calcular_propiedades_seguras

    def generar_vista_3d_simple(self, mol_data, width=630, height=350):
        """Genera vista 3D simple usando RDKit"""
        if not mol_data or not RDKIT_AVAILABLE:
            return self.crear_imagen_placeholder(width, height, "Vista 3D no disponible")
        
        try:
            mol = mol_data['mol']
            
            if mol.GetNumConformers() == 0:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except:
                    print("⚠️ No se pudo optimizar, usando conformación inicial")
            
            if not self.viewer_state['mostrar_hidrogenos']:
                mol_display = Chem.RemoveHs(mol)
            else:
                mol_display = mol
            
            conf = mol_display.GetConformer()
            coords = []
            
            for i in range(mol_display.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            coords = np.array(coords)
            
            # Aplicar rotaciones del viewer_state
            rotx = np.radians(self.viewer_state['rotacion']['x'])
            roty = np.radians(self.viewer_state['rotacion']['y'])
            rotz = np.radians(self.viewer_state['rotacion']['z'])
            
            Rx = np.array([[1, 0, 0],
                        [0, np.cos(rotx), -np.sin(rotx)],
                        [0, np.sin(rotx), np.cos(rotx)]])
            
            Ry = np.array([[np.cos(roty), 0, np.sin(roty)],
                        [0, 1, 0],
                        [-np.sin(roty), 0, np.cos(roty)]])
            
            Rz = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                        [np.sin(rotz), np.cos(rotz), 0],
                        [0, 0, 1]])
            
            coords = coords @ Rx @ Ry @ Rz
            
            zoom = self.viewer_state['zoom']
            scale = 20 * zoom
            
            x_proj = coords[:, 0] * scale + width/2
            y_proj = coords[:, 1] * scale + height/2
            z_proj = coords[:, 2]
            
            img = Image.new('RGB', (width, height), color=(20, 20, 40))
            draw = ImageDraw.Draw(img)
            
            estilo = self.viewer_state['estilo_enlace']
            
            # Dibujar enlaces
            if estilo != 'sphere':
                for bond in mol_display.GetBonds():
                    atom1_idx = bond.GetBeginAtomIdx()
                    atom2_idx = bond.GetEndAtomIdx()
                    
                    x1, y1 = int(x_proj[atom1_idx]), int(y_proj[atom1_idx])
                    x2, y2 = int(x_proj[atom2_idx]), int(y_proj[atom2_idx])
                    
                    depth_avg = (z_proj[atom1_idx] + z_proj[atom2_idx]) / 2
                    intensity = int(255 * (0.5 + depth_avg * 0.1))
                    intensity = max(100, min(255, intensity))
                    
                    if estilo == 'stick':
                        line_width = 3
                    elif estilo == 'line':
                        line_width = 1
                    else:
                        line_width = 2
                    
                    draw.line([(x1, y1), (x2, y2)], fill=(intensity, intensity, intensity), width=line_width)
            
            # Dibujar átomos
            for i in range(mol_display.GetNumAtoms()):
                atom = mol_display.GetAtomWithIdx(i)
                symbol = atom.GetSymbol()
                
                x, y = int(x_proj[i]), int(y_proj[i])
                
                # Color por elemento (CPK standard)
                if symbol == 'C':
                    color = (64, 64, 64)
                elif symbol == 'O':
                    color = (255, 50, 50)
                elif symbol == 'N':
                    color = (50, 50, 255)
                elif symbol == 'S':
                    color = (255, 255, 50)
                elif symbol == 'P':
                    color = (255, 165, 0)
                elif symbol == 'F':
                    color = (144, 224, 80)
                elif symbol == 'Cl':
                    color = (31, 240, 31)
                elif symbol == 'Br':
                    color = (166, 41, 41)
                elif symbol == 'H':
                    color = (255, 255, 255)
                else:
                    color = (200, 200, 200)
                
                depth_factor = 1 + z_proj[i] * 0.1
                
                if estilo == 'sphere':
                    radius = max(8, int(12 * depth_factor * zoom))
                elif estilo == 'stick':
                    radius = max(4, int(8 * depth_factor * zoom))
                else:
                    radius = max(2, int(4 * depth_factor * zoom))
                
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=color, outline=(255,255,255), width=1)
                
                show_label = False
                if estilo == 'line':
                    show_label = True
                elif symbol != 'C':
                    show_label = True
                elif atom.GetDegree() <= 1:
                    show_label = True
                
                if show_label and radius > 3:
                    text_x = x - len(symbol) * 3
                    text_y = y - 6
                    
                    if estilo == 'line':
                        draw.rectangle([text_x-2, text_y-2, text_x+len(symbol)*6+2, text_y+10], 
                                    fill=(0,0,0,128), outline=(255,255,255))
                    
                    draw.text((text_x, text_y), symbol, fill=(255, 255, 255))
            
            info_text = f"Estilo: {estilo.capitalize()}"
            if self.viewer_state['mostrar_hidrogenos']:
                info_text += " + H"
            
            draw.text((10, height-25), info_text, fill=(255, 255, 255))
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"❌ Error generando vista 3D: {e}")
            return self.crear_imagen_placeholder(width, height, f"Error 3D: {str(e)[:30]}")

    # Mantener métodos de visualización originales...
    def generar_imagen_2d(self, mol_data, width=630, height=350):
        """Genera imagen 2D de la molécula con RDKit - MÉTODO ORIGINAL"""
        if not mol_data or not RDKIT_AVAILABLE:
            return self.crear_imagen_placeholder(width, height, "RDKit no disponible")
            
        try:
            mol = mol_data['mol']
            
            if not mol.GetNumConformers():
                AllChem.Compute2DCoords(mol)
            
            drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
            drawer.SetFontSize(0.8)
            
            opts = drawer.drawOptions()
            opts.addStereoAnnotation = True
            opts.addAtomIndices = False
            
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            img_data = drawer.GetDrawingText()
            img_base64 = base64.b64encode(img_data).decode()
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"❌ Error generando imagen 2D: {e}")
            return self.crear_imagen_placeholder(width, height, f"Error: {str(e)[:50]}")
    
    # ... [MANTENER TODOS LOS OTROS MÉTODOS ORIGINALES] ...
    
    def crear_imagen_placeholder(self, width=630, height=350, texto="Sin molécula"):
        """Crea imagen placeholder cuando no se puede renderizar - MÉTODO ORIGINAL"""
        try:
            img = Image.new('RGB', (width, height), color=(30, 30, 50))
            draw = ImageDraw.Draw(img)
            
            try:
                font_size = 16
                bbox = draw.textbbox((0, 0), texto)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                
                draw.text((x, y), texto, fill=(255, 255, 255))
            except:
                draw.text((width//4, height//2), texto, fill=(255, 255, 255))
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"❌ Error creando placeholder: {e}")
            return None
    
    def calcular_docking_score(self, mol1_data, mol2_data):
        """Calcula score de docking simplificado entre dos moléculas - MÉTODO ORIGINAL"""
        if not mol1_data or not mol2_data or not RDKIT_AVAILABLE:
            return 0.0
            
        try:
            mol1 = mol1_data['mol']
            mol2 = mol2_data['mol']
            
            prop1 = mol1_data['propiedades']
            prop2 = mol2_data['propiedades']
            
            hbd_hba_complement = min(prop1['hbd'], prop2['hba']) + min(prop1['hba'], prop2['hbd'])
            size_complement = 50 - abs(prop1['peso_molecular'] - prop2['peso_molecular']) / 10
            logp_complement = 25 - abs(prop1['logp'] - prop2['logp']) * 5
            
            final_score = max(0, hbd_hba_complement * 10 + size_complement + logp_complement)
            
            return min(100, max(-50, final_score))
            
        except Exception as e:
            print(f"❌ Error calculando docking score: {e}")
            return 0.0

# Instancia global del visualizador
molecular_viewer = MolecularViewer()

# ===== DEFINICIÓN DE MOLÉCULAS PREDEFINIDAS (ORIGINAL) =====
MOLECULAS_PREDEFINIDAS = {
    'estradiol': {
        'smiles': 'C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O',
        'nombre': 'Estradiol',
        'descripcion': 'Hormona sexual femenina'
    },
    'fulvestrant': {
        'smiles': 'C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@H](C(F)(F)C(F)(F)S(=O)CCCCCCCCC)C2',
        'nombre': 'Fulvestrant',
        'descripcion': 'Medicamento'
    }
}

# ===== RESTO DEL CÓDIGO ORIGINAL (gestos_estado, procesamiento, etc.) =====
gestos_estado = {
    'modelo_estado': {
        'rotacion': {'x': 0, 'y': 0, 'z': 0},
        'posicion': {'x': 0, 'y': 0, 'z': 0},
        'escala': 1.0,
        'modo_gesto': 'libre',
        'manos_detectadas': 0
    },
    'molecular_data': {
        'ligando': None,
        'receptor': None,
        'docking_score': 0.0,
        'propiedades_ligando': {},
        'propiedades_receptor': {}
    }
}

ultimo_envio_websocket = 0
clientes_conectados = 0

# Mantener toda la funcionalidad original de procesamiento de gestos...
def procesar_gestos_molecular(frame):
    """Procesa gestos y actualiza el estado molecular - FUNCIÓN ORIGINAL"""
    global gestos_estado, ultimo_envio_websocket, clientes_conectados
    
    frame_flip = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
    resultado = hands.process(img_rgb)
    
    h, w, _ = frame_flip.shape
    num_manos = 0
    
    gestos_estado['modelo_estado']['manos_detectadas'] = 0
    
    if resultado.multi_hand_landmarks:
        num_manos = len(resultado.multi_hand_landmarks)
        gestos_estado['modelo_estado']['manos_detectadas'] = num_manos
        
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(frame_flip, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if num_manos == 1:
            hand = resultado.multi_hand_landmarks[0]
            palm = hand.landmark[9]
            palm_x, palm_y = palm.x, palm.y
            
            if detectar_gesto_agarre(hand.landmark, w, h):
                gestos_estado['modelo_estado']['modo_gesto'] = "rotacion_molecular"
                
                molecular_viewer.viewer_state['rotacion']['y'] = (palm_x - 0.5) * 720
                molecular_viewer.viewer_state['rotacion']['x'] = (palm_y - 0.5) * 360
                
                gestos_estado['modelo_estado']['rotacion'] = molecular_viewer.viewer_state['rotacion']
                
            else:
                gestos_estado['modelo_estado']['modo_gesto'] = "libre"
        
        elif num_manos == 2:
            hand1, hand2 = resultado.multi_hand_landmarks[0], resultado.multi_hand_landmarks[1]
            palm1, palm2 = hand1.landmark[9], hand2.landmark[9]
            
            dist = math.sqrt((palm1.x - palm2.x)**2 + (palm1.y - palm2.y)**2)
            
            agarre1 = detectar_gesto_agarre(hand1.landmark, w, h)
            agarre2 = detectar_gesto_agarre(hand2.landmark, w, h)
            
            if agarre1 and agarre2:
                gestos_estado['modelo_estado']['modo_gesto'] = "traslacion_molecular"
                
                centro_x = (palm1.x + palm2.x) / 2
                centro_y = (palm1.y + palm2.y) / 2
                
                molecular_viewer.viewer_state['traslacion']['x'] = (centro_x - 0.5) * 4
                molecular_viewer.viewer_state['traslacion']['y'] = (0.5 - centro_y) * 4
                
            elif dist > 0.3:
                gestos_estado['modelo_estado']['modo_gesto'] = "zoom_molecular"
                
                zoom_factor = min(max(dist / 0.4, 0.3), 3.0)
                molecular_viewer.viewer_state['zoom'] = zoom_factor
                gestos_estado['modelo_estado']['escala'] = zoom_factor
                
                x1, y1 = int(palm1.x * w), int(palm1.y * h)
                x2, y2 = int(palm2.x * w), int(palm2.y * h)
                cv2.line(frame_flip, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
            else:
                gestos_estado['modelo_estado']['modo_gesto'] = "cambio_vista"
        
        else:
            gestos_estado['modelo_estado']['modo_gesto'] = "libre"
    else:
        gestos_estado['modelo_estado']['modo_gesto'] = "libre"
    
    # Calcular docking score si hay dos moléculas
    if molecular_viewer.mol_ligando and molecular_viewer.mol_receptor:
        score = molecular_viewer.calcular_docking_score(
            molecular_viewer.mol_ligando, 
            molecular_viewer.mol_receptor
        )
        gestos_estado['molecular_data']['docking_score'] = score
    
    # Dibujar información en frame
    cv2.putText(frame_flip, f"Manos: {num_manos}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_flip, f"Modo: {gestos_estado['modelo_estado']['modo_gesto']}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if gestos_estado['molecular_data']['docking_score'] > 0:
        cv2.putText(frame_flip, f"Score: {gestos_estado['molecular_data']['docking_score']:.1f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Estado de RDKit
    rdkit_status = "RDKit OK" if RDKIT_AVAILABLE else "RDKit NO"
    cv2.putText(frame_flip, rdkit_status, (10, h-30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if RDKIT_AVAILABLE else (0, 0, 255), 2)
    
    # Estado del modo de vista
    modo_vista = molecular_viewer.viewer_state['modo_vista'].upper()
    cv2.putText(frame_flip, f"Vista: {modo_vista}", (10, h-60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Enviar datos via WebSocket
    tiempo_actual = time.time()
    if clientes_conectados > 0 and tiempo_actual - ultimo_envio_websocket > 0.1:
        try:
            socketio.emit('molecular_update', gestos_estado)
            ultimo_envio_websocket = tiempo_actual
        except Exception as e:
            print(f"❌ Error emitiendo WebSocket molecular: {e}")
    
    return frame_flip

def generar_frames_molecular():
    """Generador de frames con procesamiento molecular - FUNCIÓN ORIGINAL"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se puede abrir la cámara")
        for i in range(1, 4):
            print(f"🔄 Intentando cámara índice {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Cámara encontrada en índice {i}")
                break
        else:
            print("❌ No se encontró ninguna cámara disponible")
            return
    
    print("✅ Cámara abierta para visualización molecular")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Error leyendo frame {frame_count}")
            break
        
        try:
            frame_procesado = procesar_gestos_molecular(frame)
            
            _, buffer = cv2.imencode('.jpg', frame_procesado)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            
        except Exception as e:
            print(f"❌ Error procesando frame {frame_count}: {e}")
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    print("🔄 Cámara liberada")

# ===== RUTAS FLASK ORIGINALES + NUEVAS RUTAS GEMINI =====

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camara')
def camara():
    return Response(generar_frames_molecular(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cargar_molecula', methods=['POST'])
def cargar_molecula():
    """Cargar molécula desde SMILES o nombre predefinido - RUTA ORIGINAL"""
    
    if not RDKIT_AVAILABLE:
        return jsonify({'error': 'RDKit no está disponible. Instálalo con: pip install rdkit'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
        
        tipo = data.get('tipo', 'ligando')
        
        if 'smiles' in data:
            mol_data = molecular_viewer.cargar_molecula_desde_smiles(
                data['smiles'], 
                data.get('nombre', 'Molecula personalizada')
            )
        elif 'predefinida' in data:
            mol_info = MOLECULAS_PREDEFINIDAS.get(data['predefinida'])
            if not mol_info:
                return jsonify({'error': 'Molécula predefinida no encontrada'}), 400
            
            mol_data = molecular_viewer.cargar_molecula_desde_smiles(
                mol_info['smiles'],
                mol_info['nombre']
            )
        else:
            return jsonify({'error': 'Debe proporcionar SMILES o nombre predefinido'}), 400
        
        if not mol_data:
            return jsonify({'error': 'No se pudo cargar la molécula. Verifica el SMILES.'}), 400
        
        if tipo == 'ligando':
            molecular_viewer.mol_ligando = mol_data
            gestos_estado['molecular_data']['propiedades_ligando'] = mol_data['propiedades']
        else:
            molecular_viewer.mol_receptor = mol_data
            gestos_estado['molecular_data']['propiedades_receptor'] = mol_data['propiedades']
        
        if molecular_viewer.viewer_state['modo_vista'] == '2d':
            imagen = molecular_viewer.generar_imagen_2d(mol_data)
        else:
            imagen = molecular_viewer.generar_vista_3d_simple(mol_data)
        
        print(f"✅ {tipo} cargado: {mol_data['propiedades']['nombre']}")
        
        return jsonify({
            'success': True,
            'propiedades': mol_data['propiedades'],
            'smiles': mol_data['smiles'],
            'imagen_2d': imagen,
            'tipo': tipo
        })
        
    except Exception as e:
        error_msg = f"Error cargando molécula: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

# ===== NUEVAS RUTAS PARA INTEGRACIÓN GEMINI =====

@app.route('/api/gemini/analyze_molecule', methods=['POST'])
def gemini_analyze_molecule():
    """Nueva ruta: Análisis molecular usando Gemini AI"""
    try:
        data = request.get_json()
        tipo = data.get('tipo', 'ligando')
        modelo = data.get('modelo', 'gemini-1.5-flash')
        
        # Obtener datos de la molécula
        if tipo == 'ligando':
            mol_data = molecular_viewer.mol_ligando
        else:
            mol_data = molecular_viewer.mol_receptor
        
        if not mol_data:
            return jsonify({
                'error': True,
                'message': f'No hay {tipo} cargado para analizar'
            }), 400
        
        # Llamar a Gemini para análisis
        print(f"🤖 Analizando {tipo} con Gemini ({modelo})...")
        result = gemini_analyzer.analyze_molecule(mol_data, modelo)
        
        if result['error']:
            return jsonify(result), 500
        
        # Guardar análisis en el estado
        molecular_viewer.ai_state['last_analysis'] = result['analysis']
        molecular_viewer.ai_state['analysis_available'] = True
        
        print(f"✅ Análisis Gemini completado para {tipo}")
        
        return jsonify({
            'success': True,
            'analysis': result['analysis'],
            'molecule_type': tipo,
            'model_used': modelo
        })
        
    except Exception as e:
        error_msg = f"Error en análisis Gemini: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'error': True,
            'message': error_msg
        }), 500

@app.route('/api/gemini/analyze_interaction', methods=['POST'])
def gemini_analyze_interaction():
    """Nueva ruta: Análisis de interacción molecular usando Gemini AI"""
    try:
        data = request.get_json()
        modelo = data.get('modelo', 'gemini-1.5-flash')
        
        # Verificar que hay ligando y receptor
        if not molecular_viewer.mol_ligando or not molecular_viewer.mol_receptor:
            return jsonify({
                'error': True,
                'message': 'Necesita cargar tanto ligando como receptor para análisis de interacción'
            }), 400
        
        # Obtener docking score actual
        docking_score = molecular_viewer.calcular_docking_score(
            molecular_viewer.mol_ligando,
            molecular_viewer.mol_receptor
        )
        
        print(f"🤖 Analizando interacción molecular con Gemini ({modelo})...")
        
        # Llamar a Gemini para análisis de interacción
        result = gemini_analyzer.analyze_molecular_interaction(
            molecular_viewer.mol_ligando,
            molecular_viewer.mol_receptor,
            docking_score,
            modelo
        )
        
        if result['error']:
            return jsonify(result), 500
        
        # Guardar análisis en el estado
        molecular_viewer.ai_state['last_interaction_analysis'] = result['interaction_analysis']
        
        print("✅ Análisis de interacción Gemini completado")
        
        return jsonify({
            'success': True,
            'interaction_analysis': result['interaction_analysis'],
            'docking_score': docking_score,
            'model_used': modelo
        })
        
    except Exception as e:
        error_msg = f"Error en análisis de interacción: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'error': True,
            'message': error_msg
        }), 500

@app.route('/api/gemini/gesture_suggestions', methods=['POST'])
def gemini_gesture_suggestions():
    """Nueva ruta: Sugerencias contextuales basadas en gestos"""
    try:
        data = request.get_json()
        modelo = data.get('modelo', 'gemini-2.5-flash-lite')
        
        # Obtener estado actual
        gesture_mode = gestos_estado['modelo_estado']['modo_gesto']
        
        # Determinar molécula activa (priorizar ligando)
        active_molecule = molecular_viewer.mol_ligando or molecular_viewer.mol_receptor
        
        print(f"🤖 Generando sugerencias gestuales con Gemini ({modelo})...")
        
        # Llamar a Gemini para sugerencias
        result = gemini_analyzer.generate_gesture_suggestions(
            gesture_mode,
            active_molecule,
            modelo
        )
        
        if result['error']:
            return jsonify(result), 500
        
        # Guardar sugerencias en el estado
        molecular_viewer.ai_state['current_suggestions'] = result['suggestions']
        
        print("✅ Sugerencias gestuales generadas")
        
        return jsonify({
            'success': True,
            'suggestions': result['suggestions'],
            'gesture_mode': gesture_mode,
            'model_used': modelo
        })
        
    except Exception as e:
        error_msg = f"Error generando sugerencias: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'error': True,
            'message': error_msg
        }), 500

@app.route('/api/gemini/models')
def get_gemini_models():
    """Nueva ruta: Obtener modelos Gemini disponibles"""
    return jsonify({
        'success': True,
        'models': list(GEMINI_MODELS.keys()),
        'current_model': molecular_viewer.ai_state['selected_model']
    })

@app.route('/api/gemini/set_model', methods=['POST'])
def set_gemini_model():
    """Nueva ruta: Configurar modelo Gemini a usar"""
    try:
        data = request.get_json()
        modelo = data.get('modelo')
        
        if modelo not in GEMINI_MODELS:
            return jsonify({
                'error': True,
                'message': f'Modelo no válido. Opciones: {list(GEMINI_MODELS.keys())}'
            }), 400
        
        molecular_viewer.ai_state['selected_model'] = modelo
        
        return jsonify({
            'success': True,
            'model': modelo,
            'message': f'Modelo cambiado a {modelo}'
        })
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error configurando modelo: {str(e)}'
        }), 500

# ===== RUTAS ORIGINALES MANTENIDAS =====

@app.route('/api/generar_imagen')
def generar_imagen():
    """Generar imagen actualizada de la molécula - RUTA ORIGINAL"""
    if not RDKIT_AVAILABLE:
        return jsonify({'error': 'RDKit no disponible'}), 500
    
    tipo = request.args.get('tipo', 'ligando')
    modo = request.args.get('modo', molecular_viewer.viewer_state['modo_vista'])
    
    mol_data = molecular_viewer.mol_ligando if tipo == 'ligando' else molecular_viewer.mol_receptor
    
    if not mol_data:
        return jsonify({'error': f'No hay {tipo} cargado'}), 400
    
    try:
        if modo == '2d':
            imagen = molecular_viewer.generar_imagen_2d(mol_data)
        else:
            imagen = molecular_viewer.generar_vista_3d_simple(mol_data)
        
        if not imagen:
            return jsonify({'error': 'No se pudo generar la imagen'}), 500
        
        return jsonify({
            'success': True,
            'imagen': imagen,
            'viewer_state': molecular_viewer.viewer_state,
            'modo': modo
        })
        
    except Exception as e:
        error_msg = f"Error generando imagen: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/cambiar_modo_vista', methods=['POST'])
def cambiar_modo_vista():
    """Cambiar entre modo 2D y 3D - RUTA ORIGINAL"""
    if not RDKIT_AVAILABLE:
        return jsonify({'error': 'RDKit no disponible'}), 500
    
    try:
        data = request.get_json()
        nuevo_modo = data.get('modo', '2d')
        
        if nuevo_modo not in ['2d', '3d']:
            return jsonify({'error': 'Modo inválido. Use "2d" o "3d"'}), 400
        
        molecular_viewer.viewer_state['modo_vista'] = nuevo_modo
        
        print(f"🎨 Modo de vista cambiado a: {nuevo_modo}")
        
        return jsonify({
            'success': True,
            'modo': nuevo_modo,
            'viewer_state': molecular_viewer.viewer_state
        })
        
    except Exception as e:
        error_msg = f"Error cambiando modo: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/moleculas_predefinidas')
def get_moleculas_predefinidas():
    """Obtener lista de moléculas predefinidas - RUTA ORIGINAL"""
    return jsonify(MOLECULAS_PREDEFINIDAS)

@app.route('/api/reset_viewer')
def reset_viewer():
    """Reset del visualizador molecular - RUTA ORIGINAL"""
    global molecular_viewer, gestos_estado
    
    try:
        molecular_viewer.viewer_state = {
            'rotacion': {'x': 0, 'y': 0, 'z': 0},
            'traslacion': {'x': 0, 'y': 0, 'z': 0},
            'zoom': 1.0,
            'modo_vista': '3d',
            'colores': 'cpk',
            'mostrar_hidrogenos': False,
            'estilo_enlace': 'stick'
        }
        
        molecular_viewer.mol_ligando = None
        molecular_viewer.mol_receptor = None
        
        # ===== NUEVO: Reset estado AI =====
        molecular_viewer.ai_state = {
            'last_analysis': None,
            'last_interaction_analysis': None,
            'current_suggestions': None,
            'selected_model': 'gemini-1.5-flash',
            'analysis_available': False
        }
        
        gestos_estado['modelo_estado'] = {
            'rotacion': {'x': 0, 'y': 0, 'z': 0},
            'posicion': {'x': 0, 'y': 0, 'z': 0},
            'escala': 1.0,
            'modo_gesto': 'libre',
            'manos_detectadas': 0
        }
        
        gestos_estado['molecular_data'] = {
            'ligando': None,
            'receptor': None,
            'docking_score': 0.0,
            'propiedades_ligando': {},
            'propiedades_receptor': {}
        }
        
        print("🔄 Viewer y estado AI reseteados correctamente")
        return jsonify({'success': True, 'modo_vista': '3d'})
        
    except Exception as e:
        error_msg = f"Error reseteando viewer: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

# ===== WEBSOCKET EVENTS ORIGINALES + EXTENSIONES GEMINI =====

@socketio.on('connect')
def handle_connect():
    global clientes_conectados
    clientes_conectados += 1
    print(f'✅ Cliente molecular conectado. Total: {clientes_conectados}')
    emit('molecular_update', gestos_estado)

@socketio.on('disconnect') 
def handle_disconnect():
    global clientes_conectados
    clientes_conectados = max(0, clientes_conectados - 1)
    print(f'❌ Cliente molecular desconectado. Total: {clientes_conectados}')

@socketio.on('cambiar_estilo')
def handle_cambiar_estilo(data):
    """Cambiar estilo de visualización - EVENTO ORIGINAL"""
    try:
        actualizado = False
        
        if 'estilo_enlace' in data:
            molecular_viewer.viewer_state['estilo_enlace'] = data['estilo_enlace']
            actualizado = True
            print(f"🎨 Estilo de enlace cambiado a: {data['estilo_enlace']}")
            
        if 'colores' in data:
            molecular_viewer.viewer_state['colores'] = data['colores']
            actualizado = True
            print(f"🎨 Esquema de colores cambiado a: {data['colores']}")
            
        if 'mostrar_hidrogenos' in data:
            molecular_viewer.viewer_state['mostrar_hidrogenos'] = data['mostrar_hidrogenos']
            actualizado = True
            estado_h = "mostrar" if data['mostrar_hidrogenos'] else "ocultar"
            print(f"🎨 Hidrógenos: {estado_h}")
        
        if actualizado:
            emit('viewer_state_update', molecular_viewer.viewer_state, broadcast=True)
            print(f"📡 Estado del viewer actualizado y enviado a clientes")
        
    except Exception as e:
        print(f"❌ Error cambiando estilo: {e}")

@socketio.on('regenerar_imagen')
def handle_regenerar_imagen(data):
    """Regenerar imagen con estilos actualizados - EVENTO ORIGINAL"""
    try:
        tipo = data.get('tipo', 'ligando')
        emit('regenerar_molecula', {'tipo': tipo}, broadcast=True)
        print(f"🔄 Solicitando regeneración de imagen para {tipo}")
        
    except Exception as e:
        print(f"❌ Error regenerando imagen: {e}")

# ===== NUEVOS EVENTOS WEBSOCKET PARA GEMINI =====

@socketio.on('request_ai_analysis')
def handle_request_ai_analysis(data):
    """Nuevo evento: Solicitar análisis AI en tiempo real"""
    try:
        tipo = data.get('tipo', 'ligando')
        modelo = data.get('modelo', 'gemini-1.5-flash')
        
        # Obtener molécula
        mol_data = molecular_viewer.mol_ligando if tipo == 'ligando' else molecular_viewer.mol_receptor
        
        if not mol_data:
            emit('ai_analysis_error', {
                'error': f'No hay {tipo} cargado para analizar'
            })
            return
        
        # Emitir estado de "analizando"
        emit('ai_analysis_start', {
            'tipo': tipo,
            'modelo': modelo
        })
        
        print(f"🤖 Análisis AI solicitado via WebSocket: {tipo} con {modelo}")
        
        # Nota: En producción, esto debería ser asíncrono
        # Aquí solo emitimos la confirmación de que se inició
        emit('ai_analysis_queued', {
            'tipo': tipo,
            'modelo': modelo,
            'message': 'Análisis en cola, usa la API REST para obtener resultados'
        })
        
    except Exception as e:
        emit('ai_analysis_error', {
            'error': f'Error procesando solicitud: {str(e)}'
        })

@socketio.on('request_gesture_suggestions')
def handle_request_gesture_suggestions(data):
    """Nuevo evento: Solicitar sugerencias gestuales en tiempo real"""
    try:
        modelo = data.get('modelo', 'gemini-2.5-flash-lite')
        
        # Obtener modo de gesto actual
        gesture_mode = gestos_estado['modelo_estado']['modo_gesto']
        active_molecule = molecular_viewer.mol_ligando or molecular_viewer.mol_receptor
        
        emit('gesture_suggestions_start', {
            'gesture_mode': gesture_mode,
            'modelo': modelo
        })
        
        print(f"🤖 Sugerencias gestuales solicitadas via WebSocket: modo {gesture_mode}")
        
        # En producción esto sería asíncrono
        emit('gesture_suggestions_queued', {
            'gesture_mode': gesture_mode,
            'modelo': modelo,
            'message': 'Sugerencias en cola, usa la API REST para obtener resultados'
        })
        
    except Exception as e:
        emit('gesture_suggestions_error', {
            'error': f'Error generando sugerencias: {str(e)}'
        })

# ===== RUTA PARA TESTING GEMINI =====

@app.route('/test_gemini')
def test_gemini():
    """Endpoint para probar integración con Gemini"""
    try:
        # Test básico de conectividad
        test_prompt = "Responde solo 'OK' si puedes procesar este mensaje."
        result = gemini_analyzer._call_gemini_api(test_prompt, 'gemini-1.5-flash-latest')
        
        return jsonify({
            'gemini_available': result is not None,
            'api_key_configured': GEMINI_API_KEY != "TU_API_KEY_AQUI",
            'rdkit_available': RDKIT_AVAILABLE,
            'models_available': list(GEMINI_MODELS.keys()),
            'test_response': result[:100] if result else None,
            'integration_status': 'OK' if result else 'Error en conexión'
        })
        
    except Exception as e:
        return jsonify({
            'gemini_available': False,
            'error': str(e),
            'integration_status': 'Error'
        })

# ===== RUTA ORIGINAL DE TESTING =====
@app.route('/test')
def test_rdkit():
    """Endpoint para probar RDKit - RUTA ORIGINAL"""
    if not RDKIT_AVAILABLE:
        return jsonify({
            'rdkit_available': False,
            'error': 'RDKit no está instalado',
            'install_command': 'pip install rdkit'
        })
    
    try:
        test_smiles = 'CCO'
        mol = Chem.MolFromSmiles(test_smiles)
        
        if mol is None:
            return jsonify({
                'rdkit_available': True,
                'test_passed': False,
                'error': 'No se pudo crear molécula de prueba'
            })
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(200, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img_data = drawer.GetDrawingText()
        img_base64 = base64.b64encode(img_data).decode()
        
        return jsonify({
            'rdkit_available': True,
            'test_passed': True,
            'test_molecule': 'Etanol (CCO)',
            'molecular_weight': round(mw, 2),
            'logp': round(logp, 2),
            'image_generated': True,
            'image_size': len(img_base64),
            'camera_available': cv2.VideoCapture(0).isOpened(),
            'viewer_state': molecular_viewer.viewer_state,
            # ===== NUEVO: Estado Gemini =====
            'gemini_integration': True,
            'gemini_models': list(GEMINI_MODELS.keys())
        })
        
    except Exception as e:
        return jsonify({
            'rdkit_available': True,
            'test_passed': False,
            'error': str(e)
        })

# ===== RUTA PARA FAVICON =====
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    print("🧪 Iniciando visualizador molecular con RDKit + Gemini AI...")
    print("=" * 60)
    
    # Verificar RDKit
    if not RDKIT_AVAILABLE:
        print("❌ RDKit no está disponible. Por favor instálalo:")
        print("   OPCIÓN 1: pip install rdkit")
        print("   OPCIÓN 2: conda install -c conda-forge rdkit")
        print("   OPCIÓN 3: pip install rdkit-pypi")
        print("")
    else:
        print("✅ RDKit disponible")
        print("📋 Moléculas predefinidas disponibles:")
        for key, mol in MOLECULAS_PREDEFINIDAS.items():
            print(f"   - {key}: {mol['nombre']}")
        print("")
    
    # Verificar Gemini
    if GEMINI_API_KEY and GEMINI_API_KEY != "TU_API_KEY_AQUI":
        print("✅ Gemini AI configurado")
        print("🤖 Modelos disponibles:")
        for model in GEMINI_MODELS.keys():
            print(f"   - {model}")
        print("")
    else:
        print("⚠️ Gemini AI no configurado (API Key faltante)")
        print("   Configura GEMINI_API_KEY en el código")
        print("")
    
    # Verificar cámara
    cap_test = cv2.VideoCapture(0)
    if cap_test.isOpened():
        print("✅ Cámara disponible")
        cap_test.release()
    else:
        print("⚠️ Cámara no detectada - el sistema funcionará sin gestos")
    
    print("=" * 60)
    print("🌐 Servidor iniciando en http://localhost:5000")
    print("🔬 Endpoint de prueba RDKit: http://localhost:5000/test")
    print("🤖 Endpoint de prueba Gemini: http://localhost:5000/test_gemini")
    print("📹 Stream de cámara: http://localhost:5000/camara")
    print("🎨 Modo por defecto: Vista 3D")
    print("")
    print("🔥 NUEVAS FUNCIONALIDADES GEMINI:")
    print("   📊 Análisis molecular: POST /api/gemini/analyze_molecule")
    print("   🔗 Análisis interacción: POST /api/gemini/analyze_interaction")
    print("   👋 Sugerencias gestuales: POST /api/gemini/gesture_suggestions")
    print("   ⚙️  Configurar modelo: POST /api/gemini/set_model")
    print("=" * 60)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)