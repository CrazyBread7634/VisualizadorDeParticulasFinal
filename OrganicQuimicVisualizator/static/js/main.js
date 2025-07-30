// Archivo: static/js/main.js
// JavaScript principal del sistema de visualización molecular con RDKit + Gestos + IA

// ===== CONFIGURACIÓN GLOBAL =====
const socket = io();

// Variables globales del sistema
let molecularData = {
    ligando: null,
    receptor: null,
    docking_score: 0.0
};

let viewerState = {
    modo_vista: '3d',
    estilo_enlace: 'stick',
    zoom: 1.0,
    rotacion: {x: 0, y: 0, z: 0}
};

let currentMoleculeType = 'ligando';

// ===== NUEVO: Estado para IA Gemini =====
let geminiState = {
    available: false,
    currentModel: 'gemini-1.5-flash',
    lastAnalysis: null,
    lastInteractionAnalysis: null,
    analyzing: false
};

// ===== EVENTOS SOCKET.IO =====
socket.on('connect', function() {
    console.log('✅ Conectado al servidor RDKit');
    updateConnectionStatus(true);
    checkGeminiStatus(); // Verificar estado de IA
});

socket.on('disconnect', function() {
    console.log('❌ Desconectado del servidor RDKit');
    updateConnectionStatus(false);
    updateGeminiStatus(false);
});

socket.on('molecular_update', function(data) {
    console.log('📡 Update molecular:', data);
    
    if (data.modelo_estado) {
        updateGestureInfo(data.modelo_estado);
        updateViewerState(data.modelo_estado);
    }
    
    if (data.molecular_data) {
        molecularData = data.molecular_data;
        updateDockingScore();
        updateMolecularAnalysis();
    }
});

// ===== FUNCIONES DE ESTADO Y CONEXIÓN =====
function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connectionStatus');
    const textEl = document.getElementById('connectionText');
    
    if (statusEl && textEl) {
        if (connected) {
            statusEl.className = 'status-indicator status-connected';
            textEl.textContent = 'RDKit Conectado';
        } else {
            statusEl.className = 'status-indicator status-disconnected';
            textEl.textContent = 'Desconectado';
        }
    }
}

async function checkGeminiStatus() {
    try {
        const response = await fetch('/test_gemini');
        const data = await response.json();
        
        geminiState.available = data.gemini_available;
        updateGeminiStatus(data.gemini_available);
        
        console.log('🤖 Estado Gemini:', data);
        
    } catch (error) {
        console.error('❌ Error verificando Gemini:', error);
        geminiState.available = false;
        updateGeminiStatus(false);
    }
}

function updateGeminiStatus(available) {
    const statusEl = document.getElementById('geminiStatus');
    const textEl = document.getElementById('geminiText');
    
    if (statusEl && textEl) {
        if (available) {
            statusEl.className = 'status-indicator status-connected';
            textEl.textContent = 'IA: Gemini Conectado';
            geminiState.available = true;
        } else {
            statusEl.className = 'status-indicator status-disconnected';
            textEl.textContent = 'IA: No Disponible';
            geminiState.available = false;
        }
    }
}

// ===== FUNCIONES DE ACTUALIZACIÓN DE UI =====
function updateGestureInfo(gestureData) {
    const gestureModeEl = document.getElementById('gestureMode');
    const handCountEl = document.getElementById('handCount');
    const aiStatusEl = document.getElementById('aiSuggestionStatus');
    
    if (gestureModeEl) {
        gestureModeEl.textContent = gestureData.modo_gesto || 'libre';
    }
    
    if (handCountEl) {
        handCountEl.textContent = gestureData.manos_detectadas || 0;
    }
    
    // Actualizar estado de sugerencias IA
    if (aiStatusEl) {
        if (geminiState.available) {
            if (gestureData.modo_gesto !== 'libre') {
                aiStatusEl.textContent = 'Analizando...';
                aiStatusEl.style.color = '#FFD93D';
            } else {
                aiStatusEl.textContent = 'Disponible';
                aiStatusEl.style.color = '#4ECDC4';
            }
        } else {
            aiStatusEl.textContent = 'No disponible';
            aiStatusEl.style.color = '#FF6B6B';
        }
    }
}

function updateViewerState(stateData) {
    const rotationXEl = document.getElementById('rotationX');
    const rotationYEl = document.getElementById('rotationY');
    const zoomLevelEl = document.getElementById('zoomLevel');
    
    if (rotationXEl) {
        rotationXEl.textContent = `${Math.round(stateData.rotacion?.x || 0)}°`;
    }
    
    if (rotationYEl) {
        rotationYEl.textContent = `${Math.round(stateData.rotacion?.y || 0)}°`;
    }
    
    if (zoomLevelEl) {
        zoomLevelEl.textContent = `${(stateData.escala || 1.0).toFixed(1)}x`;
    }
    
    // Actualizar imagen si hay cambios significativos
    if (currentMoleculeType && hasSignificantChange(stateData)) {
        actualizarImagenMolecula();
    }
}

function hasSignificantChange(newState) {
    const threshold = 5; // grados
    const oldState = viewerState.rotacion;
    
    return Math.abs((newState.rotacion?.x || 0) - oldState.x) > threshold ||
           Math.abs((newState.rotacion?.y || 0) - oldState.y) > threshold ||
           Math.abs((newState.escala || 1.0) - viewerState.zoom) > 0.1;
}

function updateDockingScore() {
    const scoreEl = document.getElementById('dockingScore');
    if (!scoreEl) return;
    
    const score = molecularData.docking_score || 0.0;
    scoreEl.textContent = score.toFixed(1);
    
    // Cambiar color según el score
    if (score > 70) {
        scoreEl.style.color = '#4CAF50'; // Verde
    } else if (score > 40) {
        scoreEl.style.color = '#FF9800'; // Naranja
    } else {
        scoreEl.style.color = '#f44336'; // Rojo
    }
}

function updateMolecularAnalysis() {
    // Actualizar propiedades del ligando
    if (molecularData.propiedades_ligando) {
        const props = molecularData.propiedades_ligando;
        const ligandoMWEl = document.getElementById('ligandoMW');
        if (ligandoMWEl) {
            ligandoMWEl.textContent = props.peso_molecular?.toFixed(1) || '-';
        }
        updateMolecularProperties('ligando', props);
    }
    
    // Actualizar propiedades del receptor
    if (molecularData.propiedades_receptor) {
        const props = molecularData.propiedades_receptor;
        const receptorMWEl = document.getElementById('receptorMW');
        if (receptorMWEl) {
            receptorMWEl.textContent = props.peso_molecular?.toFixed(1) || '-';
        }
        updateMolecularProperties('receptor', props);
    }
    
    // Calcular análisis de interacción
    calculateInteractionAnalysis();
}

function updateMolecularProperties(tipo, propiedades) {
    const container = document.getElementById(`${tipo}Props`);
    if (!container) return;
    
    container.style.display = 'grid';
    container.innerHTML = `
        <div class="property-item">
            <div class="property-value">${propiedades.peso_molecular?.toFixed(1) || '-'}</div>
            <div>MW (Da)</div>
        </div>
        <div class="property-item">
            <div class="property-value">${propiedades.logp?.toFixed(2) || '-'}</div>
            <div>LogP</div>
        </div>
        <div class="property-item">
            <div class="property-value">${propiedades.hbd || '-'}</div>
            <div>H-Donors</div>
        </div>
        <div class="property-item">
            <div class="property-value">${propiedades.hba || '-'}</div>
            <div>H-Acceptors</div>
        </div>
        <div class="property-item">
            <div class="property-value">${propiedades.tpsa?.toFixed(1) || '-'}</div>
            <div>TPSA</div>
        </div>
        <div class="property-item">
            <div class="property-value">${propiedades.rotatable_bonds || '-'}</div>
            <div>Rot. Bonds</div>
        </div>
    `;
}

function calculateInteractionAnalysis() {
    const ligando = molecularData.propiedades_ligando;
    const receptor = molecularData.propiedades_receptor;
    
    if (!ligando || !receptor) return;
    
    // Cálculos simplificados de análisis
    const similarity = Math.min(100, (molecularData.docking_score + 50) * 100 / 150);
    const complementarity = Math.min(ligando.hbd, receptor.hba) + Math.min(ligando.hba, receptor.hbd);
    const avgLogP = (ligando.logp + receptor.logp) / 2;
    const hbondPotential = complementarity;
    
    // Actualizar elementos UI
    const similarityEl = document.getElementById('similarity');
    const complementarityEl = document.getElementById('complementarity');
    const lipophilicityEl = document.getElementById('lipophilicity');
    const hbondingEl = document.getElementById('hbonding');
    
    if (similarityEl) similarityEl.textContent = `${similarity.toFixed(0)}%`;
    if (complementarityEl) complementarityEl.textContent = `${(complementarity * 10).toFixed(0)}%`;
    if (lipophilicityEl) lipophilicityEl.textContent = avgLogP.toFixed(2);
    if (hbondingEl) hbondingEl.textContent = hbondPotential;
}

// ===== FUNCIONES DE CARGA DE MOLÉCULAS =====
async function cargarMoleculaPredefinida(tipo) {
    const selectorId = `${tipo}Selector`;
    const selector = document.getElementById(selectorId);
    if (!selector) return;
    
    const moleculeName = selector.value;
    if (!moleculeName) return;
    
    try {
        const response = await fetch('/api/cargar_molecula', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                tipo: tipo,
                predefinida: moleculeName
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            console.log(`✅ ${tipo} cargado:`, data.propiedades.nombre);
            mostrarImagenMolecula(data.imagen_2d);
            currentMoleculeType = tipo;
            updateMolecularProperties(tipo, data.propiedades);
            
            // Notificar a IA sobre nueva molécula
            if (typeof geminiAI !== 'undefined' && geminiAI) {
                geminiAI.onMoleculeLoaded(tipo, data);
            }
        } else {
            console.error('❌ Error cargando molécula:', data.error);
            alert('Error cargando molécula: ' + data.error);
        }
        
    } catch (error) {
        console.error('❌ Error en la petición:', error);
        alert('Error de conexión al cargar molécula');
    }
}

async function cargarMoleculaSmiles(tipo) {
    const smilesId = `${tipo}Smiles`;
    const smilesInput = document.getElementById(smilesId);
    if (!smilesInput) return;
    
    const smiles = smilesInput.value.trim();
    
    if (!smiles) {
        alert('Por favor ingresa un SMILES válido');
        return;
    }
    
    try {
        const response = await fetch('/api/cargar_molecula', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                tipo: tipo,
                smiles: smiles,
                nombre: `${tipo.charAt(0).toUpperCase() + tipo.slice(1)} personalizado`
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            console.log(`✅ ${tipo} SMILES cargado:`, smiles);
            mostrarImagenMolecula(data.imagen_2d);
            currentMoleculeType = tipo;
            updateMolecularProperties(tipo, data.propiedades);
            
            // Limpiar el campo de SMILES
            smilesInput.value = '';
            
            // Notificar a IA sobre nueva molécula
            if (typeof geminiAI !== 'undefined' && geminiAI) {
                geminiAI.onMoleculeLoaded(tipo, data);
            }
        } else {
            console.error('❌ Error cargando SMILES:', data.error);
            alert('Error cargando SMILES: ' + data.error);
        }
        
    } catch (error) {
        console.error('❌ Error en la petición SMILES:', error);
        alert('Error de conexión al cargar SMILES');
    }
}

// ===== FUNCIONES DE VISUALIZACIÓN =====
function mostrarImagenMolecula(imagenBase64) {
    if (!imagenBase64) return;
    
    const loadingDisplay = document.getElementById('loadingDisplay');
    const moleculeImage = document.getElementById('moleculeImage');
    
    if (loadingDisplay) loadingDisplay.style.display = 'none';
    if (moleculeImage) {
        moleculeImage.style.display = 'block';
        moleculeImage.src = imagenBase64;
    }
}

async function actualizarImagenMolecula() {
    if (!currentMoleculeType) return;
    
    try {
        const response = await fetch(`/api/generar_imagen?tipo=${currentMoleculeType}&modo=${viewerState.modo_vista}`);
        const data = await response.json();
        
        if (data.success && data.imagen) {
            mostrarImagenMolecula(data.imagen);
            if (data.viewer_state) {
                viewerState = {...viewerState, ...data.viewer_state};
            }
            const renderStyleEl = document.getElementById('renderStyle');
            if (renderStyleEl) {
                renderStyleEl.textContent = viewerState.estilo_enlace;
            }
        }
        
    } catch (error) {
        console.error('❌ Error actualizando imagen:', error);
    }
}

// ===== FUNCIONES DE CONTROL DE VISTA =====
async function cambiarVista(modo) {
    try {
        // Actualizar estado local
        viewerState.modo_vista = modo;
        
        // Enviar cambio al servidor
        const response = await fetch('/api/cambiar_modo_vista', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({modo: modo})
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Actualizar botones
            const btn2D = document.getElementById('btn2D');
            const btn3D = document.getElementById('btn3D');
            
            if (btn2D) btn2D.classList.remove('active');
            if (btn3D) btn3D.classList.remove('active');
            
            const activeBtn = document.getElementById(`btn${modo.toUpperCase()}`);
            if (activeBtn) activeBtn.classList.add('active');
            
            // Actualizar imagen
            await actualizarImagenMolecula();
            
            console.log(`🎨 Vista cambiada a: ${modo}`);
        } else {
            console.error('❌ Error cambiando vista:', data.error);
        }
        
    } catch (error) {
        console.error('❌ Error en cambiarVista:', error);
    }
}

function cambiarEstilo(estilo) {
    console.log(`🎨 Cambiando estilo a: ${estilo}`);
    
    // Actualizar estado local
    viewerState.estilo_enlace = estilo;
    
    // Enviar cambio via WebSocket
    socket.emit('cambiar_estilo', {
        estilo_enlace: estilo
    });
    
    // Actualizar botones visuales
    document.querySelectorAll('.control-btn').forEach(btn => {
        if (btn.textContent.toLowerCase().includes(estilo.toLowerCase())) {
            btn.classList.add('active');
        } else if (['stick', 'sphere', 'line'].some(s => btn.textContent.toLowerCase().includes(s))) {
            btn.classList.remove('active');
        }
    });
    
    // Regenerar imagen inmediatamente
    socket.emit('regenerar_imagen', {
        tipo: currentMoleculeType
    });
}

function toggleHydrogens() {
    console.log('🔄 Cambiando visibilidad de hidrógenos');
    
    // Cambiar estado
    viewerState.mostrar_hidrogenos = !viewerState.mostrar_hidrogenos;
    
    // Enviar cambio via WebSocket
    socket.emit('cambiar_estilo', {
        mostrar_hidrogenos: viewerState.mostrar_hidrogenos
    });
    
    // Actualizar botón visual
    const btnH = document.querySelector('.control-btn:nth-child(6)'); // Botón H atoms
    if (btnH) {
        if (viewerState.mostrar_hidrogenos) {
            btnH.classList.add('active');
            btnH.textContent = 'H atoms ✓';
        } else {
            btnH.classList.remove('active');
            btnH.textContent = 'H atoms';
        }
    }
    
    // Regenerar imagen inmediatamente
    socket.emit('regenerar_imagen', {
        tipo: currentMoleculeType
    });
}

// ===== FUNCIONES DE SISTEMA =====
async function resetViewer() {
    try {
        const response = await fetch('/api/reset_viewer');
        const data = await response.json();
        
        if (data.success) {
            // Reset UI original
            const loadingDisplay = document.getElementById('loadingDisplay');
            const moleculeImage = document.getElementById('moleculeImage');
            const ligandoProps = document.getElementById('ligandoProps');
            const receptorProps = document.getElementById('receptorProps');
            
            if (loadingDisplay) loadingDisplay.style.display = 'flex';
            if (moleculeImage) moleculeImage.style.display = 'none';
            if (ligandoProps) ligandoProps.style.display = 'none';
            if (receptorProps) receptorProps.style.display = 'none';
            
            // Reset selectors
            const ligandoSelector = document.getElementById('ligandoSelector');
            const receptorSelector = document.getElementById('receptorSelector');
            const ligandoSmiles = document.getElementById('ligandoSmiles');
            const receptorSmiles = document.getElementById('receptorSmiles');
            
            if (ligandoSelector) ligandoSelector.value = '';
            if (receptorSelector) receptorSelector.value = '';
            if (ligandoSmiles) ligandoSmiles.value = '';
            if (receptorSmiles) receptorSmiles.value = '';
            
            // Reset análisis
            const dockingScore = document.getElementById('dockingScore');
            const ligandoMW = document.getElementById('ligandoMW');
            const receptorMW = document.getElementById('receptorMW');
            
            if (dockingScore) dockingScore.textContent = '0.0';
            if (ligandoMW) ligandoMW.textContent = '-';
            if (receptorMW) receptorMW.textContent = '-';
            
            // Reset botones de vista
            const btn2D = document.getElementById('btn2D');
            const btn3D = document.getElementById('btn3D');
            
            if (btn2D) btn2D.classList.remove('active');
            if (btn3D) btn3D.classList.add('active'); // 3D por defecto
            
            // Reset botones de estilo
            document.querySelectorAll('.control-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Reset estado IA
            geminiState.lastAnalysis = null;
            geminiState.lastInteractionAnalysis = null;
            
            const aiResults = document.getElementById('ai-analysis-results');
            if (aiResults) {
                aiResults.innerHTML = `
                    <div class="analysis-placeholder">
                        🤖 Carga una molécula y haz clic en "Analizar Molécula" para obtener un análisis detallado con IA de Gemini.
                        <br><br>
                        💡 <strong>Funciones disponibles:</strong>
                        <br>• Análisis químico profesional
                        <br>• Evaluación de interacciones moleculares  
                        <br>• Sugerencias contextuales por gestos
                    </div>
                `;
            }
            
            // Reset estado viewer
            viewerState = {
                modo_vista: '3d',
                estilo_enlace: 'stick',
                zoom: 1.0,
                rotacion: {x: 0, y: 0, z: 0},
                mostrar_hidrogenos: false
            };
            
            currentMoleculeType = null;
            console.log('🔄 Viewer y IA reseteados completamente');
        } else {
            console.error('❌ Error en reset:', data.error);
        }
        
    } catch (error) {
        console.error('❌ Error reseteando viewer:', error);
    }
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

// ===== FUNCIONES DE EXPORTACIÓN =====
async function exportarMolecula(tipo) {
    try {
        const data = molecularData[`propiedades_${tipo}`];
        if (!data) {
            alert(`No hay ${tipo} cargado para exportar`);
            return;
        }
        
        const exportData = {
            tipo: tipo,
            propiedades: data,
            timestamp: new Date().toISOString()
        };
        
        downloadJSON(exportData, `${tipo}_${Date.now()}.json`);
        console.log(`📄 ${tipo} exportado`);
        
    } catch (error) {
        console.error('❌ Error exportando:', error);
    }
}

async function exportarAnalisis() {
    try {
        const analysisData = {
            docking_score: molecularData.docking_score,
            ligando: molecularData.propiedades_ligando,
            receptor: molecularData.propiedades_receptor,
            viewer_state: viewerState,
            interaction_analysis: {
                similarity: document.getElementById('similarity')?.textContent || '0%',
                complementarity: document.getElementById('complementarity')?.textContent || '0%',
                lipophilicity: document.getElementById('lipophilicity')?.textContent || '0.0',
                hbonding: document.getElementById('hbonding')?.textContent || '0'
            },
            timestamp: new Date().toISOString()
        };
        
        downloadJSON(analysisData, `molecular_analysis_${Date.now()}.json`);
        console.log('📊 Análisis exportado');
        
    } catch (error) {
        console.error('❌ Error exportando análisis:', error);
    }
}

async function exportarAnalisisIA() {
    try {
        if (!geminiState.lastAnalysis && !geminiState.lastInteractionAnalysis) {
            alert('No hay análisis de IA disponibles para exportar. Realiza un análisis primero.');
            return;
        }
        
        const exportData = {
            timestamp: new Date().toISOString(),
            gemini_model: geminiState.currentModel,
            molecular_analysis: geminiState.lastAnalysis,
            interaction_analysis: geminiState.lastInteractionAnalysis,
            molecular_data: molecularData,
            viewer_state: viewerState
        };
        
        downloadJSON(exportData, `gemini_analysis_${Date.now()}.json`);
        console.log('🤖 Análisis IA exportado');
        
    } catch (error) {
        console.error('❌ Error exportando análisis IA:', error);
        alert('Error exportando análisis de IA');
    }
}

function exportData() {
    exportarAnalisis();
}

// ===== FUNCIONES DE UTILIDAD =====
function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ===== FUNCIONES DE TESTING =====
async function testGemini() {
    console.log('🤖 Probando conexión con Gemini...');
    
    try {
        const response = await fetch('/test_gemini');
        const data = await response.json();
        
        if (data.gemini_available) {
            alert(`✅ Gemini AI conectado correctamente!\n\nModelos disponibles: ${data.models_available?.join(', ') || 'N/A'}\nEstado: ${data.integration_status}`);
        } else {
            alert(`❌ Gemini AI no disponible.\nError: ${data.error || 'Desconocido'}\nVerifica la configuración de API Key.`);
        }
        
        updateGeminiStatus(data.gemini_available);
        
    } catch (error) {
        console.error('❌ Error en test Gemini:', error);
        alert(`❌ Error probando Gemini: ${error.message}`);
        updateGeminiStatus(false);
    }
}

// ===== FUNCIONES GLOBALES PARA INTEGRACIÓN CON IA =====
window.analyzeCurrentMolecule = function(type = 'ligando') {
    if (typeof geminiAI !== 'undefined' && geminiAI) {
        return geminiAI.analyzeMolecule(type);
    } else {
        console.error('❌ Gemini AI no está inicializado');
    }
};

window.analyzeCurrentInteraction = function() {
    if (typeof geminiAI !== 'undefined' && geminiAI) {
        return geminiAI.analyzeInteraction();
    } else {
        console.error('❌ Gemini AI no está inicializado');
    }
};

window.getCurrentGestureSuggestions = function() {
    if (typeof geminiAI !== 'undefined' && geminiAI) {
        return geminiAI.getGestureSuggestions();
    } else {
        console.error('❌ Gemini AI no está inicializado');
    }
};

// ===== INICIALIZACIÓN =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('🧪 RDKit Molecular Viewer + IA iniciado');
    updateConnectionStatus(false);
    
    // Verificar estado inicial de Gemini
    setTimeout(() => {
        checkGeminiStatus();
    }, 1000);
    
    // Cargar moléculas por defecto para demo
    setTimeout(() => {
        const ligandoSelector = document.getElementById('ligandoSelector');
        if (ligandoSelector) {
            ligandoSelector.value = 'estradiol';
            cargarMoleculaPredefinida('ligando');
        }
        
        setTimeout(() => {
            const receptorSelector = document.getElementById('receptorSelector');
            if (receptorSelector) {
                receptorSelector.value = 'fulvestrant';
                cargarMoleculaPredefinida('receptor');
            }
        }, 1000);
    }, 2000);
});

// ===== OBJETO DEBUG GLOBAL =====
window.rdkitDebug = {
    molecularData,
    viewerState,
    geminiState,
    socket,
    cargarMolecula: cargarMoleculaPredefinida,
    cambiarVista,
    reset: resetViewer,
    testGemini,
    analyzeWithAI: window.analyzeCurrentMolecule,
    exportAI: exportarAnalisisIA
};

console.log('🧪 RDKit Molecular Viewer + 🤖 Gemini AI JavaScript cargado');
console.log('🔧 Debug disponible en: window.rdkitDebug');