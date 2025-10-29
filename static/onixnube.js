document.addEventListener('DOMContentLoaded', () => {
    const sendBtn = document.getElementById('sendBtn');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const fileInput = document.getElementById('fileInput');
    const fileIndicator = document.getElementById('file-indicator');
    const menuBtn = document.getElementById('menu-btn');
    const sidebar = document.getElementById('sidebar');
    const chatHistoryContainer = document.getElementById('chat-history-container');
    const newChatBtn = document.querySelector('.new-chat-btn');
    const tempChatBtn = document.querySelector('.temp-chat-btn');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModalOverlay = document.getElementById('settings-modal-overlay');
    const instructionsTextarea = document.getElementById('instructions-textarea');
    const saveInstructionsBtn = document.getElementById('save-instructions-btn');
    const closeInstructionsBtn = document.getElementById('close-instructions-btn');
    const searchModeBtn = document.getElementById('search-mode-btn');
    const imageGenModeBtn = document.getElementById('image-gen-mode-btn');

    let API_BASE = localStorage.getItem('onixnube_api_base') || '';
    if (!API_BASE) {
        const loc = window.location;
        if (/ngrok/.test(loc.hostname)) {
            API_BASE = `${loc.protocol}//${loc.host}`;
        } else {
            API_BASE = '';
        }
    }
    function setApiBase(newBase){
        API_BASE = newBase.replace(/\/$/, '');
        localStorage.setItem('onixnube_api_base', API_BASE);
        showNotification(`API Base configurada: ${API_BASE || '(relativa)'}`, 'info');
    }
    window.setOnixApiBase = setApiBase; 

    const API_URL = (path => () => `${API_BASE}${path}`)('/chat');
    const PDF_API_URL = (path => () => `${API_BASE}${path}`)('/generate_pdf');
    const HEALTH_URL = (path => () => `${API_BASE}${path}`)('/health');
    const MAX_RETRY_ATTEMPTS = 1;
    const REQUEST_TIMEOUT = 190000;
    const QUICK_MODE_TIMEOUT = 15000;

    let attachedFile = null;
    let abortController = null;
    let conversations = [];
    let activeConversationId = null;
    let isTempChat = false;
    let searchModeActive = false;
    let imageGenModeActive = false;
    let quickModeActive = false;
    let isProcessing = false;
    let sidebarOverlay = null;
    let quickModeBtn = null;
    let quickModeIndicator = null;
    let serverStatusElement = null;

    async function init() {
        createUIElements();
    await loadRemoteConfig();
    await checkServerHealth();
        loadConversations();
        loadInstructions();
        setupEventListeners();
        autoResizeTextarea();
        showWelcomeMessage();
        detectDevice();
        setupKeyboardShortcuts();
    }

    function createUIElements() {
        sidebarOverlay = document.createElement('div');
        sidebarOverlay.className = 'sidebar-overlay';
        document.body.appendChild(sidebarOverlay);
        
        quickModeBtn = document.createElement('button');
        quickModeBtn.className = 'action-btn quick-mode-btn';
        quickModeBtn.innerHTML = '<i class="fas fa-bolt"></i> <span>Modo Rápido</span>';
        quickModeBtn.title = 'Respuestas ultra-rápidas sin contexto (Ctrl+Q)';
        
        const actionButtonsContainer = document.querySelector('.action-buttons-container');
        actionButtonsContainer.appendChild(quickModeBtn);
        
        quickModeIndicator = document.createElement('div');
        quickModeIndicator.className = 'quick-mode-indicator';
        quickModeIndicator.innerHTML = '<i class="fas fa-bolt"></i> Modo Rápido Activo';
        
        const chatInputContainer = document.querySelector('.chat-input-container');
        chatInputContainer.insertBefore(quickModeIndicator, chatInputContainer.firstChild);
        
        serverStatusElement = document.createElement('div');
        serverStatusElement.className = 'server-status';
        serverStatusElement.innerHTML = '<div class="status-dot"></div> <span>Nube AI</span>';
        document.querySelector('.navbar').appendChild(serverStatusElement);
    }

    function detectDevice() {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const isTouch = 'ontouchstart' in window;
        
        document.body.classList.toggle('mobile-device', isMobile);
        document.body.classList.toggle('touch-device', isTouch);
        
        if (isMobile) {
            sidebar.classList.add('hidden');
        }
    }

    async function checkServerHealth() {
        try {
            const response = await fetch(HEALTH_URL(), { 
                signal: AbortSignal.timeout(3000) 
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('✅ Nube AI:', data.status);
                updateServerStatus(true);
                return true;
            }
        } catch (error) {
            console.warn('⚠️ Conectando con Nube AI...');
            updateServerStatus(false);
            
            setTimeout(checkServerHealth, 5000);
            return false;
        }
    }

    async function loadRemoteConfig(){
        try {
            const res = await fetch(`${API_BASE}/config`, { signal: AbortSignal.timeout(3000) });
            if(res.ok){
                const cfg = await res.json();
                console.log('Config remota:', cfg);
            }
        } catch(e){
            console.warn('No se pudo cargar /config', e.message);
        }
    }

    function updateServerStatus(online) {
        if (serverStatusElement) {
            const dot = serverStatusElement.querySelector('.status-dot');
            const text = serverStatusElement.querySelector('span');
            
            if (online) {
                dot.style.background = 'var(--success-color)';
                text.textContent = 'Nube AI';
            } else {
                dot.style.background = 'var(--warning-color)';
                text.textContent = 'Conectando...';
            }
        }
    }

    function setupEventListeners() {
        searchModeBtn.addEventListener('click', toggleSearchMode);
        imageGenModeBtn.addEventListener('click', toggleImageGenMode);
        quickModeBtn.addEventListener('click', toggleQuickMode);

        sendBtn.addEventListener('click', handleSend);
        userInput.addEventListener('keydown', handleKeyPress);
        userInput.addEventListener('input', () => {
            autoResizeTextarea();
            handleTyping();
        });
        userInput.addEventListener('paste', handlePaste);

        menuBtn.addEventListener('click', toggleSidebar);
        sidebarOverlay.addEventListener('click', closeSidebar);
        newChatBtn.addEventListener('click', () => createNewConversation(true));
        tempChatBtn.addEventListener('click', startTempChat);

        settingsBtn.addEventListener('click', openSettings);
        saveInstructionsBtn.addEventListener('click', saveInstructions);
        closeInstructionsBtn.addEventListener('click', closeSettings);
        settingsModalOverlay.addEventListener('click', (e) => {
            if (e.target === settingsModalOverlay) closeSettings();
        });

        fileInput.addEventListener('change', handleFileUpload);

        window.addEventListener('resize', debounce(handleResize, 250));
        window.addEventListener('online', () => {
            showNotification('Conexión restaurada', 'success');
            checkServerHealth();
        });
        window.addEventListener('offline', () => {
            showNotification('Sin conexión', 'error');
            updateServerStatus(false);
        });
        
        document.addEventListener('gesturestart', (e) => e.preventDefault());
    }

    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'q') {
                e.preventDefault();
                toggleQuickMode();
            }
            
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                createNewConversation(true);
            }
            
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                toggleSidebar();
            }
            
            if (e.key === 'Escape') {
                if (!settingsModalOverlay.classList.contains('hidden')) {
                    closeSettings();
                }
                if (window.innerWidth <= 768 && !sidebar.classList.contains('hidden')) {
                    closeSidebar();
                }
            }
        });
    }

    function toggleSearchMode() {
        searchModeActive = !searchModeActive;
        searchModeBtn.classList.toggle('active', searchModeActive);
        
        if (searchModeActive) {
            imageGenModeActive = false;
            imageGenModeBtn.classList.remove('active');
        }
        
        updateInputState();
        userInput.focus();
    }

    function toggleImageGenMode() {
        imageGenModeActive = !imageGenModeActive;
        imageGenModeBtn.classList.toggle('active', imageGenModeActive);
        
        if (imageGenModeActive) {
            searchModeActive = false;
            searchModeBtn.classList.remove('active');
        }
        
        updateInputState();
        userInput.focus();
    }

    function toggleQuickMode() {
        quickModeActive = !quickModeActive;
        quickModeBtn.classList.toggle('active', quickModeActive);
        quickModeIndicator.classList.toggle('active', quickModeActive);
        document.querySelector('.chat-input').classList.toggle('quick-mode', quickModeActive);
        sendBtn.classList.toggle('quick-mode', quickModeActive);
        
        if (quickModeActive) {
            showNotification('⚡ Modo Rápido: Sin contexto, máxima velocidad', 'success');
            clearFile();
            fileInput.disabled = true;
            const uploadBtn = document.querySelector('.upload-btn');
            if (uploadBtn) {
                uploadBtn.style.opacity = '0.5';
                uploadBtn.style.pointerEvents = 'none';
                uploadBtn.title = 'Adjuntos deshabilitados en Modo Rápido';
            }
        } else {
            fileInput.disabled = false;
            const uploadBtn = document.querySelector('.upload-btn');
            if (uploadBtn) {
                uploadBtn.style.opacity = '';
                uploadBtn.style.pointerEvents = '';
                uploadBtn.title = 'Adjuntar archivo';
            }
        }
        
        updateInputState();
        userInput.focus();
    }

    function updateInputState() {
        let placeholder = "Escribe tu mensaje...";
        
        if (quickModeActive) {
            placeholder = "⚡ Modo rápido - Escribe tu pregunta...";
        } else if (imageGenModeActive) {
            placeholder = "Describe la imagen que quieres crear...";
        } else if (searchModeActive) {
            placeholder = "Buscaré información actualizada...";
        }
        
        userInput.placeholder = placeholder;
    }

    function loadConversations() {
        const stored = localStorage.getItem('onixnube_conversations');
        conversations = stored ? JSON.parse(stored) : [];
        
        const weekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
        conversations = conversations.filter(c => {
            const timestamp = parseInt(c.id.split('_')[1]);
            return timestamp > weekAgo;
        });
        
        if (conversations.length > 30) {
            conversations = conversations.slice(0, 30);
        }
        
        const lastActiveId = localStorage.getItem('onixnube_active_id');
        if (lastActiveId && conversations.find(c => c.id === lastActiveId)) {
            activeConversationId = lastActiveId;
        } else if (conversations.length > 0) {
            activeConversationId = conversations[0].id;
        }
        
        renderSidebar();
        if (activeConversationId) {
            loadChat(activeConversationId);
        }
    }

    function saveState() {
        if (isTempChat) return;
        
        localStorage.setItem('onixnube_conversations', JSON.stringify(conversations));
        if (activeConversationId) {
            localStorage.setItem('onixnube_active_id', activeConversationId);
        }
    }

    function renderSidebar() {
        chatHistoryContainer.innerHTML = '';
        
        if (conversations.length === 0 && !isTempChat) {
            chatHistoryContainer.innerHTML = '<p class="no-history">No hay conversaciones</p>';
            return;
        }
        
        const fragment = document.createDocumentFragment();
        
        conversations.forEach(convo => {
            const el = document.createElement('div');
            el.className = 'chat-history-item';
            el.dataset.id = convo.id;
            
            const title = escapeHtml(convo.title);
            const truncatedTitle = title.length > 30 ? title.substring(0, 30) + '...' : title;
            
            el.innerHTML = `
                <span><i class="far fa-comment-dots"></i> ${truncatedTitle}</span>
                <div class="actions">
                    <i class="fas fa-trash-alt delete-btn"></i>
                </div>
            `;
            
            if (convo.id === activeConversationId && !isTempChat) {
                el.classList.add('active');
            }
            
            el.addEventListener('click', (e) => {
                if (e.target.classList.contains('delete-btn')) {
                    e.stopPropagation();
                    deleteConversation(convo.id);
                } else {
                    selectConversation(convo.id);
                }
            });
            
            fragment.appendChild(el);
        });
        
        chatHistoryContainer.appendChild(fragment);
    }

    function selectConversation(id) {
        isTempChat = false;
        loadChat(id);
        
        if (window.innerWidth <= 768) {
            closeSidebar();
        }
    }

    function loadChat(id) {
        if (!id && !isTempChat) {
            createNewConversation(false);
            return;
        }
        
        const conversation = conversations.find(c => c.id === id);
        if (conversation) {
            activeConversationId = id;
            chatMessages.innerHTML = '';
            
            let index = 0;
            function loadNextMessage() {
                if (index < conversation.messages.length) {
                    addMessageToChat(conversation.messages[index].content, conversation.messages[index].sender, false);
                    index++;
                    requestAnimationFrame(loadNextMessage);
                } else {
                    renderSidebar();
                    saveState();
                    scrollToBottom();
                }
            }
            
            requestAnimationFrame(loadNextMessage);
        }
    }

    function createNewConversation(save = true) {
        isTempChat = false;
        activeConversationId = `convo_${Date.now()}`;
        
        const newConvo = {
            id: activeConversationId,
            title: "Nueva conversación",
            messages: []
        };
        
        if (save) {
            conversations.unshift(newConvo);
            saveState();
        }
        
        chatMessages.innerHTML = '';
        showWelcomeMessage();
        renderSidebar();
        
        if (window.innerWidth <= 768) {
            closeSidebar();
        }
    }

    function startTempChat() {
        isTempChat = true;
        activeConversationId = `temp_${Date.now()}`;
        chatMessages.innerHTML = '';
        
        addMessageToChat({
            type: 'text',
            text: "🔒 **Chat Temporal**\n\nEsta conversación no se guardará."
        }, 'ia', false);
        
        renderSidebar();
        
        if (window.innerWidth <= 768) {
            closeSidebar();
        }
    }

    function deleteConversation(idToDelete) {
        if (confirm('¿Eliminar conversación?')) {
            conversations = conversations.filter(c => c.id !== idToDelete);
            
            if (activeConversationId === idToDelete) {
                activeConversationId = conversations.length > 0 ? conversations[0].id : null;
                if (activeConversationId) {
                    loadChat(activeConversationId);
                } else {
                    createNewConversation(false);
                }
            }
            
            saveState();
            renderSidebar();
        }
    }

    function loadInstructions() {
        const saved = localStorage.getItem('onixnube_instructions') || '';
        instructionsTextarea.value = saved;
    }

    function openSettings() {
        settingsModalOverlay.classList.remove('hidden');
        instructionsTextarea.focus();
    }

    function closeSettings() {
        settingsModalOverlay.classList.add('hidden');
    }

    function saveInstructions() {
        const instructions = instructionsTextarea.value.trim();
        localStorage.setItem('onixnube_instructions', instructions);
        closeSettings();
        showNotification('Instrucciones guardadas', 'success');
    }

    function toggleSidebar() {
        const isHidden = sidebar.classList.contains('hidden');
        
        if (window.innerWidth <= 768) {
            if (isHidden) {
                sidebar.classList.remove('hidden');
                sidebar.classList.add('shown');
                sidebarOverlay.classList.add('active');
            } else {
                closeSidebar();
            }
        } else {
            sidebar.classList.toggle('hidden');
        }
    }

    function closeSidebar() {
        sidebar.classList.add('hidden');
        sidebar.classList.remove('shown');
        sidebarOverlay.classList.remove('active');
    }

    async function handleFileUpload(event) {
        if (quickModeActive) {
            showNotification('El Modo Rápido no admite adjuntos', 'warning');
            fileInput.value = '';
            return;
        }
        const file = event.target.files[0];
        if (!file) return;
        
        const maxSize = 5 * 1024 * 1024; // 5MB
        if (file.size > maxSize) {
            showNotification('Archivo muy grande (máx 5MB)', 'error');
            fileInput.value = '';
            return;
        }
        
        showFileIndicator(`Procesando ${file.name}...`, true);
        
        try {
            let fileData = { name: file.name };
            
            if (file.type.startsWith('image/')) {
                fileData.type = 'image';
                fileData.content = await compressImage(file);
            } else if (file.type === 'text/plain' || file.type === 'text/rtf') {
                fileData.type = 'document';
                fileData.content = await file.text();
            } else if (file.type === 'application/pdf') {
                fileData.type = 'document';
                fileData.content = await extractTextFromPdf(file);
            } else if (file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
                fileData.type = 'document';
                fileData.content = await extractTextFromDocx(file);
            } else {
                throw new Error('Tipo no soportado');
            }
            
            attachedFile = fileData;
            showFileIndicator(file.name);
            
        } catch (error) {
            showNotification(`Error: ${error.message}`, 'error');
            clearFile();
        }
    }

    async function compressImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = (e) => {
                const img = new Image();
                img.src = e.target.result;
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    const maxDim = 800;
                    let width = img.width;
                    let height = img.height;
                    
                    if (width > height) {
                        if (width > maxDim) {
                            height *= maxDim / width;
                            width = maxDim;
                        }
                    } else {
                        if (height > maxDim) {
                            width *= maxDim / height;
                            height = maxDim;
                        }
                    }
                    
                    canvas.width = width;
                    canvas.height = height;
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    const quality = file.size > 1000000 ? 0.6 : 0.8;
                    resolve(canvas.toDataURL('image/jpeg', quality));
                };
                img.onerror = reject;
            };
            reader.onerror = reject;
        });
    }

    async function extractTextFromPdf(file) {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js';
        
        try {
            const pdf = await pdfjsLib.getDocument(URL.createObjectURL(file)).promise;
            let text = '';
            const maxPages = Math.min(pdf.numPages, 5);
            
            for (let i = 1; i <= maxPages; i++) {
                const page = await pdf.getPage(i);
                const content = await page.getTextContent();
                text += content.items.map(item => item.str).join(' ') + '\n\n';
            }
            
            return text.substring(0, 5000);
        } catch (error) {
            throw new Error('Error leyendo PDF');
        }
    }

    async function extractTextFromDocx(file) {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const result = await mammoth.extractRawText({ arrayBuffer });
            return result.value.substring(0, 5000);
        } catch (error) {
            throw new Error('Error leyendo documento');
        }
    }

    function showFileIndicator(fileName, isLoading = false) {
        fileIndicator.style.display = 'flex';
        
        let content;
        if (isLoading) {
            content = `
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
                <span>${escapeHtml(fileName)}</span>
            `;
        } else if (attachedFile && attachedFile.type === 'image') {
            content = `
                <div class="file-preview">
                    <img src="${attachedFile.content}" alt="preview">
                </div>
                <span>${escapeHtml(fileName)}</span>
                <button onclick="clearFileExternal()">×</button>
            `;
        } else {
            content = `
                <span>📄 ${escapeHtml(fileName)}</span>
                <button onclick="clearFileExternal()">×</button>
            `;
        }
        
        fileIndicator.innerHTML = content;
    }

    window.clearFileExternal = clearFile;

    function clearFile() {
        attachedFile = null;
        fileInput.value = '';
        fileIndicator.style.display = 'none';
    }

    async function handlePaste(e) {
        if (quickModeActive) {
            return;
        }
        const items = e.clipboardData.items;
        
        for (const item of items) {
            if (item.type.startsWith('image')) {
                e.preventDefault();
                const blob = item.getAsFile();
                const file = new File([blob], 'imagen-pegada.png', { type: blob.type });
                
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
                break;
            }
        }
    }

    let typingTimer;
    function handleTyping() {
        clearTimeout(typingTimer);
        typingTimer = setTimeout(() => {
        }, 1000);
    }

    async function handleSend() {
        if (isProcessing) {
            stopGeneration();
        } else {
            sendMessage();
        }
    }

    function handleKeyPress(e) {
        if (e.key === 'Enter' && !e.shiftKey && !isProcessing) {
            e.preventDefault();
            sendMessage();
        }
    }

    async function sendMessage(retryCount = 0) {
        const startTime = Date.now();
        const messageText = userInput.value.trim();
        
        if (!messageText && !attachedFile) return;
        
        const welcomeMsg = chatMessages.querySelector('.welcome-message');
        if (welcomeMsg) welcomeMsg.remove();
        
        if (!activeConversationId && !isTempChat) {
            createNewConversation();
        }
        
        const userContent = { 
            type: 'text', 
            text: messageText || `[Archivo: ${attachedFile.name}]`
        };
        
        addMessageToChat(userContent, 'user', false);
        saveMessage(userContent, 'user');
        
        userInput.value = '';
        autoResizeTextarea();
        
        const loadingText = quickModeActive 
            ? '⚡ Procesando...'
            : attachedFile 
            ? `Analizando ${attachedFile.name}...`
            : searchModeActive 
            ? 'Buscando...'
            : imageGenModeActive
            ? 'Generando imagen...'
            : 'Pensando...';
        
        const loadingElement = addMessageToChat({
            type: 'text',
            text: loadingText
        }, 'ia', false);
        
        loadingElement.classList.add('typing-cursor');
        
        isProcessing = true;
        abortController = new AbortController();
        setStopButtonState();
        
        let conversationHistory = [];
        if (!quickModeActive && activeConversationId && !isTempChat) {
            const currentConvo = conversations.find(c => c.id === activeConversationId);
            if (currentConvo && currentConvo.messages) {
                const recentMessages = currentConvo.messages.slice(-4, -1);
                conversationHistory = recentMessages.map(msg => ({
                    role: msg.sender === 'user' ? 'user' : 'assistant',
                    content: (msg.content.text || '').substring(0, 300)
                }));
            }
        }
        
        try {
            const payload = {
                mensaje: messageText,
                archivo: attachedFile,
                instrucciones: quickModeActive ? '' : (localStorage.getItem('onixnube_instructions') || ''),
                historial: quickModeActive ? [] : conversationHistory,
                modo: imageGenModeActive ? 'generate_image' : (searchModeActive ? 'search' : 'chat'),
                quick_mode: quickModeActive
            };
            
            const timeout = quickModeActive ? QUICK_MODE_TIMEOUT : REQUEST_TIMEOUT;
            const response = await Promise.race([
                fetch(API_URL(), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: abortController.signal
                }),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Timeout')), timeout)
                )
            ]);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            loadingElement.parentElement.remove();
            
            if (data.error) {
                addMessageToChat({
                    type: 'text',
                    text: `⚠️ ${data.error}`
                }, 'ia');
                
                if (!quickModeActive && data.error.includes('tardando')) {
                    showNotification('💡 Prueba el Modo Rápido para respuestas instantáneas', 'info');
                }
            } else {
                const responseContent = data.contenido || {
                    type: 'text',
                    text: 'Sin respuesta'
                };
                
                addMessageToChat(responseContent, 'ia');
                saveMessage(responseContent, 'ia');
                
                if (quickModeActive) {
                    const endTime = Date.now();
                    console.log(`⚡ Respuesta en: ${endTime - startTime}ms`);
                }
            }
            
        } catch (error) {
            loadingElement.parentElement.remove();
            
            if (error.name === 'AbortError') return;
            
            let errorMessage = '⚠️ ';
            if (error.message === 'Timeout') {
                errorMessage += quickModeActive 
                    ? 'Tiempo agotado. Intenta de nuevo.'
                    : 'Respuesta lenta. Activa el Modo Rápido.';
            } else if (!navigator.onLine) {
                errorMessage += 'Sin conexión.';
            } else {
                errorMessage += 'Error de conexión.';
            }
            
            addMessageToChat({ type: 'text', text: errorMessage }, 'ia');
            
            if (retryCount < MAX_RETRY_ATTEMPTS && navigator.onLine && !quickModeActive) {
                showNotification('Reintentando...', 'warning');
                setTimeout(() => sendMessage(retryCount + 1), 2000);
            }
            
        } finally {
            isProcessing = false;
            abortController = null;
            setSendButtonState();
            clearFile();
        }
        
    }

    function addMessageToChat(content, sender, shouldAnimate = true) {
        const messageContainer = document.createElement('div');
        messageContainer.className = sender === 'user' ? 'user-container' : 'ia-message-container';
        
        const messageElement = document.createElement('div');
        messageElement.classList.add(sender);
        
        if (content.type === 'image') {
            const img = document.createElement('img');
            img.src = content.text;
            img.className = 'ia-generated-image';
            img.loading = 'lazy';
            img.onclick = () => openImageModal(content.text);
            messageElement.appendChild(img);
        } else {
            if (shouldAnimate && sender === 'ia' && content.text.length < 500) {
                typeWriterEffect(messageElement, content.text);
            } else {
                messageElement.innerHTML = marked.parse(content.text || '');
            }
        }
        
        messageContainer.appendChild(messageElement);
        
        if (sender === 'ia') {
            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'action-buttons';
            
            if (content.type === 'image') {
                const downloadBtn = createActionButton(
                    '<i class="fas fa-download"></i> Descargar',
                    () => downloadImage(content.text)
                );
                actionsContainer.appendChild(downloadBtn);
            } else {
                const copyBtn = createActionButton(
                    '<i class="fas fa-copy"></i>',
                    () => copyToClipboard(content.text)
                );
                
                const pdfBtn = createActionButton(
                    '<i class="fas fa-file-pdf"></i>',
                    () => downloadAsPdf(messageElement.innerHTML)
                );
                
                actionsContainer.appendChild(copyBtn);
                actionsContainer.appendChild(pdfBtn);
            }
            
            messageContainer.appendChild(actionsContainer);
        }
        
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
        
        return messageElement;
    }

    function createActionButton(html, onClick) {
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.innerHTML = html;
        btn.onclick = onClick;
        return btn;
    }

    function saveMessage(content, sender) {
        if (isTempChat) return;
        
        let conversation = conversations.find(c => c.id === activeConversationId);
        
        if (!conversation) {
            activeConversationId = `convo_${Date.now()}`;
            const titleText = content.text || 'Chat';
            const title = titleText.substring(0, 35).replace(/\n/g, ' ');
            
            conversation = {
                id: activeConversationId,
                title: title,
                messages: []
            };
            
            conversations.unshift(conversation);
        }
        
        if (sender === 'user' && conversation.messages.length === 0 && content.text) {
            conversation.title = content.text.substring(0, 35).replace(/\n/g, ' ');
        }
        
        if (conversation.messages.length > 50) {
            conversation.messages = conversation.messages.slice(-30);
        }
        
        conversation.messages.push({ sender, content });
        saveState();
        renderSidebar();
    }

    function setStopButtonState() {
        sendBtn.innerHTML = '<i class="fas fa-stop"></i>';
        sendBtn.title = 'Detener';
        userInput.disabled = true;
    }


    function setSendButtonState() {
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        sendBtn.title = quickModeActive ? 'Enviar (Modo Rápido)' : 'Enviar';
        userInput.disabled = false;
        userInput.focus();
    }

    function stopGeneration() {
        if (abortController) {
            abortController.abort();
            isProcessing = false;
            setSendButtonState();
            showNotification('Generación detenida', 'info');
        }
    }

    function typeWriterEffect(element, text, speed = 15) {
        element.innerHTML = '';
        
        if (text.length > 300) {
            element.innerHTML = marked.parse(text);
            return;
        }
        
        element.classList.add('typing-cursor');
        let index = 0;
        const cleanText = text.replace(/[*_`#]/g, '');
        
        function type() {
            if (index < cleanText.length) {
                element.textContent += cleanText.charAt(index);
                index++;
                
                const nextSpeed = userInput.value ? 5 : speed;
                setTimeout(type, nextSpeed);
            } else {
                element.classList.remove('typing-cursor');
                element.innerHTML = marked.parse(text);
            }
        }
        
        type();
    }

    function showWelcomeMessage() {
        const hour = new Date().getHours();
        let greeting = 'Hola';
        
        if (hour >= 5 && hour < 12) greeting = 'Buenos días';
        else if (hour >= 12 && hour < 19) greeting = 'Buenas tardes';
        else greeting = 'Buenas noches';
        
        const welcomeHTML = `
            <div class="welcome-message" style="
                text-align: center;
                padding: 2rem 1rem;
                color: var(--text-secondary);
                animation: fadeIn 0.5s;
                max-width: 500px;
                margin: 0 auto;
            ">
                <h2 style="
                    font-size: 2rem;
                    margin-bottom: 0.75rem;
                    background: var(--accent-gradient);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">${greeting}, soy Nube</h2>
                <p style="font-size: 0.95rem; margin-bottom: 1.5rem; opacity: 0.9;">
                    Tu asistente de IA
                </p>
                <div style="
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 0.5rem;
                    margin-bottom: 1.5rem;
                ">
                    <div style="
                        padding: 0.75rem;
                        background: var(--secondary-color);
                        border-radius: 8px;
                        border: 1px solid var(--border-color);
                    ">
                        <i class="fas fa-bolt" style="font-size: 1.25rem; margin-bottom: 0.4rem; color: var(--quick-mode-color);"></i>
                        <p style="font-size: 0.75rem;">Modo Rápido</p>
                    </div>
                    <div style="
                        padding: 0.75rem;
                        background: var(--secondary-color);
                        border-radius: 8px;
                        border: 1px solid var(--border-color);
                    ">
                        <i class="fas fa-brain" style="font-size: 1.25rem; margin-bottom: 0.4rem; color: #667eea;"></i>
                        <p style="font-size: 0.75rem;">IA Inteligente</p>
                    </div>
                    <div style="
                        padding: 0.75rem;
                        background: var(--secondary-color);
                        border-radius: 8px;
                        border: 1px solid var(--border-color);
                    ">
                        <i class="fas fa-image" style="font-size: 1.25rem; margin-bottom: 0.4rem; color: #764ba2;"></i>
                        <p style="font-size: 0.75rem;">Visión</p>
                    </div>
                    <div style="
                        padding: 0.75rem;
                        background: var(--secondary-color);
                        border-radius: 8px;
                        border: 1px solid var(--border-color);
                    ">
                        <i class="fas fa-globe" style="font-size: 1.25rem; margin-bottom: 0.4rem; color: #667eea;"></i>
                        <p style="font-size: 0.75rem;">Web</p>
                    </div>
                </div>
                <p style="font-size: 0.85rem; opacity: 0.7;">
                    💡 Tip: Usa <kbd style="
                        padding: 0.2rem 0.4rem;
                        background: var(--secondary-color);
                        border-radius: 4px;
                        border: 1px solid var(--border-color);
                        font-size: 0.8rem;
                    ">Ctrl+Q</kbd> para activar el Modo Rápido
                </p>
            </div>
        `;
        
        chatMessages.innerHTML = welcomeHTML;
    }

    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    function scrollToBottom() {
        if (chatMessages) {
            requestAnimationFrame(() => {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });
        }
    }

    function autoResizeTextarea() {
        userInput.style.height = 'auto';
        const newHeight = Math.min(userInput.scrollHeight, 100);
        userInput.style.height = newHeight + 'px';
    }

    function handleResize() {
        if (window.innerWidth > 768) {
            sidebarOverlay.classList.remove('active');
        }
        
        if (window.visualViewport) {
            const viewportHeight = window.visualViewport.height;
            document.documentElement.style.setProperty('--vh', `${viewportHeight * 0.01}px`);
        }
    }

    async function copyToClipboard(text) {
        try {
            const cleanText = text.replace(/[*_`#]/g, '').trim();
            await navigator.clipboard.writeText(cleanText);
            showNotification('📋 Copiado', 'success');
        } catch (err) {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            
            try {
                document.execCommand('copy');
                showNotification('📋 Copiado', 'success');
            } catch (e) {
                showNotification('Error al copiar', 'error');
            }
            
            document.body.removeChild(textarea);
        }
    }

    function showNotification(message, type = 'info') {
        document.querySelectorAll('.notification').forEach(n => n.remove());
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        let icon = '';
        switch(type) {
            case 'success': icon = '✅'; break;
            case 'error': icon = '❌'; break;
            case 'warning': icon = '⚠️'; break;
            case 'info': icon = 'ℹ️'; break;
        }
        
        notification.innerHTML = `${icon} ${message}`;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'fadeOut 0.3s';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    async function downloadAsPdf(htmlContent) {
        try {
            showNotification('Generando PDF...', 'info');
            
            const response = await fetch(PDF_API_URL(), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ html: htmlContent })
            });
            
            if (!response.ok) throw new Error('Error generando PDF');
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `nube_chat_${new Date().toISOString().slice(0,10)}.pdf`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showNotification('PDF descargado', 'success');
        } catch (error) {
            showNotification('Error generando PDF', 'error');
        }
    }

    function downloadImage(dataUrl, filename) {
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = filename || `nube_imagen_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        showNotification('Imagen descargada', 'success');
    }

    function openImageModal(imageSrc) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 3000;
            cursor: zoom-out;
            padding: 1rem;
            animation: fadeIn 0.2s;
        `;
        
        const img = document.createElement('img');
        img.src = imageSrc;
        img.style.cssText = `
            max-width: 95%;
            max-height: 95%;
            border-radius: 10px;
            box-shadow: 0 0 30px rgba(0,0,0,0.5);
            object-fit: contain;
            animation: slideInUp 0.3s;
        `;
        
        modal.onclick = () => modal.remove();
        const handleEsc = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', handleEsc);
            }
        };
        document.addEventListener('keydown', handleEsc);
        
        modal.appendChild(img);
        document.body.appendChild(modal);
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').catch(() => {
        });
    }

    window.addEventListener('error', (e) => {
        console.error('Error global:', e);
    });

    window.addEventListener('unhandledrejection', (e) => {
        console.error('Promise rechazada:', e);
    });

    init();
});

const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-10px); }
    }
    
    kbd {
        display: inline-block;
        padding: 0.2rem 0.4rem;
        font-size: 0.8rem;
        line-height: 1;
        color: var(--text-primary);
        background-color: var(--secondary-color);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        box-shadow: inset 0 -1px 0 var(--border-color);
        font-family: monospace;
    }
    
    .welcome-message {
        opacity: 0;
        animation: fadeIn 0.5s forwards;
    }
    
    /* Performance optimizations */
    .chat-messages {
        contain: layout style;
    }
    
    .user-container,
    .ia-message-container {
        contain: layout;
    }
    
    /* Smooth scrolling en iOS */
    .chat-messages {
        -webkit-overflow-scrolling: touch;
    }
`;
document.head.appendChild(style);