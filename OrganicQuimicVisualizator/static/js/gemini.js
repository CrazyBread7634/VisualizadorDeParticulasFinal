// Archivo: static/js/gemini-integration.js
// Integración Gemini AI para el sistema de visualización molecular

class GeminiMolecularAI {
    constructor() {
        this.currentModel = 'gemini-1.5-flash';
        this.availableModels = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-2.5-flash-lite',
            'gemini-2.5-pro'
        ];
        this.analysisCache = new Map();
        this.isAnalyzing = false;
        
        this.initializeUI();
        this.setupEventListeners();
    }
    
    initializeUI() {
        // Crear panel de IA si no existe
        if (!document.getElementById('ai-panel')) {
            this.createAIPanel();
        }
        
        // Inicializar selector de modelo
        this.updateModelSelector();
    }
    
    createAIPanel() {
        // Agregar panel de IA al layout existente
        const analysisPanel = document.querySelector('.analysis-panel');
        if (!analysisPanel) return;
        
        const aiSection = document.createElement('div');
        aiSection.className = 'section';
        aiSection.id = 'ai-panel';
        aiSection.innerHTML = `
            <h3>🤖 Análisis con IA</h3>
            <div class="ai-controls">
                <select id="gemini-model-select">
                    <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
                    <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                    <option value="gemini-2.5-flash-lite">Gemini 2.5 Flash Lite</option>
                    <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                </select>
                <button id="analyze-molecule-btn" class="control-btn">
                    <span class="btn-text">Analizar Molécula</span>
                    <div class="btn-loader" style="display: none;">
                        <div class="spinner"></div>
                    </div>
                </button>
                <button id="analyze-interaction-btn" class="control-btn" disabled>
                    <span class="btn-text">Analizar Interacción</span>
                    <div class="btn-loader" style="display: none;">
                        <div class="spinner"></div>
                    </div>
                </button>
                <button id="gesture-suggestions-btn" class="control-btn">
                    <span class="btn-text">Sugerencias Gestuales</span>
                    <div class="btn-loader" style="display: none;">
                        <div class="spinner"></div>
                    </div>
                </button>
            </div>
            <div id="ai-analysis-results" class="ai-results">
                <div class="analysis-placeholder">
                    Carga una molécula y haz clic en "Analizar Molécula" para obtener un análisis detallado con IA.
                </div>
            </div>
        `;
        
        // Insertar antes de la última sección
        const lastSection = analysisPanel.lastElementChild;
        analysisPanel.insertBefore(aiSection, lastSection);
        
        // Agregar estilos CSS específicos
        this.addAIStyles();
    }
    
    addAIStyles() {
        const styles = `
            <style>
            .ai-controls {
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-bottom: 15px;
            }
            
            #gemini-model-select {
                width: 100%;
                padding: 8px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                color: white;
                font-size: 14px;
            }
            
            .control-btn {
                position: relative;
                min-height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .btn-loader {
                position: absolute;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .spinner {
                width: 20px;
                height: 20px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-top: 2px solid #4ECDC4;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .ai-results {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                max-height: 400px;
                overflow-y: auto;
            }
            
            .analysis-placeholder {
                text-align: center;
                color: rgba(255, 255, 255, 0.6);
                font-style: italic;
                line-height: 1.5;
            }
            
            .analysis-content {
                color: white;
                font-size: 14px;
                line-height: 1.6;
            }
            
            .analysis-content h4 {
                color: #4ECDC4;
                margin-top: 15px;
                margin-bottom: 8px;
                font-size: 16px;
                border-bottom: 1px solid rgba(78, 205, 196, 0.3);
                padding-bottom: 4px;
            }
            
            .analysis-content p {
                margin-bottom: 12px;
                text-align: justify;
            }
            
            .interaction-analysis {
                border-left: 3px solid #FFD93D;
                padding-left: 12px;
                margin: 10px 0;
            }
            
            .gesture-suggestions {
                border-left: 3px solid #FF6B6B;
                padding-left: 12px;
                margin: 10px 0;
            }
            
            .ai-error {
                color: #ff6b6b;
                background: rgba(255, 107, 107, 0.1);
                padding: 10px;
                border-radius: 6px;
                border: 1px solid rgba(255, 107, 107, 0.3);
            }
            
            .ai-success {
                color: #4ecdc4;
                background: rgba(78, 205, 196, 0.1);
                padding: 10px;
                border-radius: 6px;
                border: 1px solid rgba(78, 205, 196, 0.3);
            }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    setupEventListeners() {
        // Configurar eventos cuando el DOM esté listo
        document.addEventListener('DOMContentLoaded', () => {
            this.initializeEventListeners();
        });
        
        // Si ya está listo
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.initializeEventListeners();
            });
        } else {
            this.initializeEventListeners();
        }
    }
    
    initializeEventListeners() {
        const modelSelect = document.getElementById('gemini-model-select');
        const analyzeMoleculeBtn = document.getElementById('analyze-molecule-btn');
        const analyzeInteractionBtn = document.getElementById('analyze-interaction-btn');
        const gestureSuggestionsBtn = document.getElementById('gesture-suggestions-btn');
        
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.currentModel = e.target.value;
                console.log(`🤖 Modelo Gemini cambiado a: ${this.currentModel}`);
            });
        }
        
        if (analyzeMoleculeBtn) {
            analyzeMoleculeBtn.addEventListener('click', () => {
                this.analyzeMolecule();
            });
        }
        
        if (analyzeInteractionBtn) {
            analyzeInteractionBtn.addEventListener('click', () => {
                this.analyzeInteraction();
            });
        }
        
        if (gestureSuggestionsBtn) {
            gestureSuggestionsBtn.addEventListener('click', () => {
                this.getGestureSuggestions();
            });
        }
        
        // Escuchar cambios en el estado molecular para habilitar/deshabilitar botones
        if (typeof socket !== 'undefined') {
            socket.on('molecular_update', (data) => {
                this.updateButtonStates(data);
            });
        }
    }
    
    updateButtonStates(molecularData) {
        const analyzeInteractionBtn = document.getElementById('analyze-interaction-btn');
        
        if (analyzeInteractionBtn && molecularData.molecular_data) {
            const hasLigando = molecularData.molecular_data.propiedades_ligando && 
                             Object.keys(molecularData.molecular_data.propiedades_ligando).length > 0;
            const hasReceptor = molecularData.molecular_data.propiedades_receptor && 
                              Object.keys(molecularData.molecular_data.propiedades_receptor).length > 0;
            
            analyzeInteractionBtn.disabled = !(hasLigando && hasReceptor);
            
            if (hasLigando && hasReceptor) {
                analyzeInteractionBtn.title = 'Analizar interacción entre ligando y receptor';
            } else {
                analyzeInteractionBtn.title = 'Carga tanto ligando como receptor para análisis de interacción';
            }
        }
    }
    
    updateModelSelector() {
        const modelSelect = document.getElementById('gemini-model-select');
        if (modelSelect) {
            modelSelect.value = this.currentModel;
        }
    }
    
    async analyzeMolecule(tipo = 'ligando') {
        if (this.isAnalyzing) return;
        
        const btn = document.getElementById('analyze-molecule-btn');
        this.setButtonLoading(btn, true);
        this.isAnalyzing = true;
        
        try {
            const response = await fetch('/api/gemini/analyze_molecule', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    tipo: tipo,
                    modelo: this.currentModel
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayMolecularAnalysis(data.analysis, tipo);
                console.log('✅ Análisis molecular completado:', data.analysis);
            } else {
                this.displayError(data.message || 'Error en el análisis molecular');
                console.error('❌ Error en análisis:', data.message);
            }
            
        } catch (error) {
            this.displayError(`Error de conexión: ${error.message}`);
            console.error('❌ Error de red:', error);
        } finally {
            this.setButtonLoading(btn, false);
            this.isAnalyzing = false;
        }
    }
    
    async analyzeInteraction() {
        if (this.isAnalyzing) return;
        
        const btn = document.getElementById('analyze-interaction-btn');
        this.setButtonLoading(btn, true);
        this.isAnalyzing = true;
        
        try {
            const response = await fetch('/api/gemini/analyze_interaction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    modelo: this.currentModel
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayInteractionAnalysis(data.interaction_analysis, data.docking_score);
                console.log('✅ Análisis de interacción completado:', data.interaction_analysis);
            } else {
                this.displayError(data.message || 'Error en el análisis de interacción');
                console.error('❌ Error en análisis de interacción:', data.message);
            }
            
        } catch (error) {
            this.displayError(`Error de conexión: ${error.message}`);
            console.error('❌ Error de red:', error);
        } finally {
            this.setButtonLoading(btn, false);
            this.isAnalyzing = false;
        }
    }
    
    async getGestureSuggestions() {
        if (this.isAnalyzing) return;
        
        const btn = document.getElementById('gesture-suggestions-btn');
        this.setButtonLoading(btn, true);
        
        try {
            const response = await fetch('/api/gemini/gesture_suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    modelo: 'gemini-2.5-flash-lite' // Usar modelo rápido para sugerencias
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayGestureSuggestions(data.suggestions, data.gesture_mode);
                console.log('✅ Sugerencias gestuales generadas:', data.suggestions);
            } else {
                this.displayError(data.message || 'Error generando sugerencias');
                console.error('❌ Error en sugerencias:', data.message);
            }
            
        } catch (error) {
            this.displayError(`Error de conexión: ${error.message}`);
            console.error('❌ Error de red:', error);
        } finally {
            this.setButtonLoading(btn, false);
        }
    }
    
    setButtonLoading(button, isLoading) {
        if (!button) return;
        
        const textSpan = button.querySelector('.btn-text');
        const loader = button.querySelector('.btn-loader');
        
        if (isLoading) {
            button.disabled = true;
            if (textSpan) textSpan.style.opacity = '0';
            if (loader) loader.style.display = 'flex';
        } else {
            button.disabled = false;
            if (textSpan) textSpan.style.opacity = '1';
            if (loader) loader.style.display = 'none';
        }
    }
    
    displayMolecularAnalysis(analysis, tipo) {
        const resultsDiv = document.getElementById('ai-analysis-results');
        if (!resultsDiv) return;
        
        const html = `
            <div class="analysis-content">
                <div class="ai-success">
                    ✅ Análisis completado para ${tipo} con ${this.currentModel}
                </div>
                
                <h4>💊 ${analysis.suggestedName || 'Nombre Sugerido'}</h4>
                <p>${analysis.suggestedName || 'No disponible'}</p>
                
                <h4>🧪 Características Químicas Clave</h4>
                <p>${analysis.keyChemicalFeatures || 'No disponible'}</p>
                
                <h4>🎯 Propiedades Farmacológicas Potenciales</h4>
                <p>${analysis.potentialPharmacologicalProperties || 'No disponible'}</p>
                
                <h4>🏥 Posibles Usos</h4>
                <p>${analysis.potentialUses || 'No disponible'}</p>
                
                <h4>📊 Análisis de Lipinski</h4>
                <p>${analysis.lipinskiAnalysis || 'No disponible'}</p>
                
                <h4>🔬 Perspectivas Estructurales</h4>
                <p>${analysis.structuralInsights || 'No disponible'}</p>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        resultsDiv.scrollTop = 0;
        
        // Animar entrada
        resultsDiv.style.opacity = '0';
        setTimeout(() => {
            resultsDiv.style.transition = 'opacity 0.3s ease';
            resultsDiv.style.opacity = '1';
        }, 100);
    }
    
    displayInteractionAnalysis(analysis, dockingScore) {
        const resultsDiv = document.getElementById('ai-analysis-results');
        if (!resultsDiv) return;
        
        const html = `
            <div class="analysis-content">
                <div class="ai-success">
                    ✅ Análisis de interacción completado con ${this.currentModel}
                </div>
                
                <div class="interaction-analysis">
                    <h4>🔗 Análisis de Interacción (Score: ${dockingScore.toFixed(1)})</h4>
                    <p>${analysis.interactionAnalysis || 'No disponible'}</p>
                    
                    <h4>🤝 Interacciones Probables</h4>
                    <p>${analysis.probableInteractions || 'No disponible'}</p>
                    
                    <h4>📈 Interpretación del Score</h4>
                    <p>${analysis.scoreInterpretation || 'No disponible'}</p>
                    
                    <h4>🚀 Sugerencias de Optimización</h4>
                    <p>${analysis.optimizationSuggestions || 'No disponible'}</p>
                    
                    <h4>🧬 Relevancia Biológica</h4>
                    <p>${analysis.biologicalRelevance || 'No disponible'}</p>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        resultsDiv.scrollTop = 0;
        
        // Animar entrada
        resultsDiv.style.opacity = '0';
        setTimeout(() => {
            resultsDiv.style.transition = 'opacity 0.3s ease';
            resultsDiv.style.opacity = '1';
        }, 100);
    }
    
    displayGestureSuggestions(suggestions, gestureMode) {
        const resultsDiv = document.getElementById('ai-analysis-results');
        if (!resultsDiv) return;
        
        const html = `
            <div class="analysis-content">
                <div class="ai-success">
                    ✅ Sugerencias generadas para modo: ${gestureMode}
                </div>
                
                <div class="gesture-suggestions">
                    <h4>👋 Acciones con Gestos</h4>
                    <p>${suggestions.gestureActions || 'No disponible'}</p>
                    
                    <h4>🔍 Observaciones Químicas</h4>
                    <p>${suggestions.chemicalObservations || 'No disponible'}</p>
                    
                    <h4>💡 Consejos de Interacción</h4>
                    <p>${suggestions.interactionTips || 'No disponible'}</p>
                    
                    <h4>➡️ Siguientes Pasos</h4>
                    <p>${suggestions.nextSteps || 'No disponible'}</p>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        resultsDiv.scrollTop = 0;
        
        // Animar entrada
        resultsDiv.style.opacity = '0';
        setTimeout(() => {
            resultsDiv.style.transition = 'opacity 0.3s ease';
            resultsDiv.style.opacity = '1';
        }, 100);
    }
    
    displayError(message) {
        const resultsDiv = document.getElementById('ai-analysis-results');
        if (!resultsDiv) return;
        
        const html = `
            <div class="ai-error">
                ❌ Error: ${message}
            </div>
        `;
        
        resultsDiv.innerHTML = html;
    }
    
    // Método público para integrar con el sistema existente
    onMoleculeLoaded(moleculeType, moleculeData) {
        console.log(`🤖 Molécula cargada: ${moleculeType}`, moleculeData);
        
        // Auto-análisis si está habilitado
        if (this.autoAnalysis && !this.isAnalyzing) {
            setTimeout(() => {
                this.analyzeMolecule(moleculeType);
            }, 1000);
        }
    }
    
    // Método para integrar con el sistema de gestos
    onGestureChange(gestureMode) {
        console.log(`🤖 Modo de gesto cambiado: ${gestureMode}`);
        
        // Actualizar sugerencias automáticamente cada 10 segundos si hay actividad
        if (this.autoSuggestions && gestureMode !== 'libre') {
            clearTimeout(this.suggestionTimeout);
            this.suggestionTimeout = setTimeout(() => {
                this.getGestureSuggestions();
            }, 10000);
        }
    }
    
    // Configuración
    setAutoAnalysis(enabled) {
        this.autoAnalysis = enabled;
    }
    
    setAutoSuggestions(enabled) {
        this.autoSuggestions = enabled;
    }
}

// ===== INTEGRACIÓN CON EL SISTEMA EXISTENTE =====

// Instancia global
let geminiAI = null;

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    // Esperar un poco para que el sistema principal se inicialice
    setTimeout(() => {
        geminiAI = new GeminiMolecularAI();
        console.log('🤖 Gemini AI integrado correctamente');
        
        // Integrar con el sistema existente si existe
        if (typeof window.rdkitDebug !== 'undefined') {
            window.rdkitDebug.geminiAI = geminiAI;
            console.log('🔗 Gemini AI añadido a rdkitDebug');
        }
    }, 2000);
});

// Funciones globales para integración
window.GeminiMolecularAI = GeminiMolecularAI;

// Integrar con el socket existente si está disponible
if (typeof io !== 'undefined') {
    // Esperar a que el socket esté disponible
    setTimeout(() => {
        if (typeof socket !== 'undefined') {
            // Escuchar eventos moleculares para auto-análisis
            socket.on('molecular_update', function(data) {
                if (geminiAI && data.modelo_estado) {
                    geminiAI.onGestureChange(data.modelo_estado.modo_gesto);
                }
            });
        }
    }, 3000);
}

// Funciones de utilidad para integración manual
window.analyzeCurrentMolecule = function(type = 'ligando') {
    if (geminiAI) {
        return geminiAI.analyzeMolecule(type);
    } else {
        console.error('❌ Gemini AI no está inicializado');
    }
};

window.analyzeCurrentInteraction = function() {
    if (geminiAI) {
        return geminiAI.analyzeInteraction();
    } else {
        console.error('❌ Gemini AI no está inicializado');
    }
};

window.getCurrentGestureSuggestions = function() {
    if (geminiAI) {
        return geminiAI.getGestureSuggestions();
    } else {
        console.error('❌ Gemini AI no está inicializado');
    }
};

console.log('🤖 Script de integración Gemini AI cargado');