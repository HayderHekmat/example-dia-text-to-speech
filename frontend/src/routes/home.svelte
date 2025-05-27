<script lang="ts">
    import ChatInterface from '../components/ChatInterface.svelte';
    import GenerationSettings from '../components/GenerationSettings.svelte';
    import SoundEffectsPanel from '../components/SoundEffectsPanel.svelte';
    import AudioControls from '../components/AudioControls.svelte';
    import GenerationButton from '../components/GenerationButton.svelte';
    import AudioOutput from '../components/AudioOutput.svelte';
    
    interface Message {
        speaker: string;
        text: string;
    }

    // Main state
    let messages: Message[] = [];
    let currentInput = '';
    let availableSpeakers: string[] = ['[S1]'];
    let selectedSpeaker = '[S1]';
    let audioUrl: string | null = null;
    let isProcessing = false;
    let selectedEffect = '';
    let error: string | null = null;
    let isRecording = false;
    let recordedAudioUrl: string | null = null;
    let uploadedAudioUrl: string | null = null;
    let editingMessageIndex: number | null = null;
    let editText = '';
    let generationProgress = 0;

    // Generation settings
    let maxNewTokens = 1024;  
    let cfgScale = 3.0;      
    let temperature = 1.2;   
    let topP = 0.9;          
    let cfgFilterTopK = 32;   
    let speedFactor = 0.9;    

    function updateAvailableSpeakers() {
        if (messages.length === 0) {
            availableSpeakers = ['[S1]'];
            selectedSpeaker = '[S1]';
            return;
        }
        
        availableSpeakers = ['[S1]', '[S2]'];
        
        if (!availableSpeakers.includes(selectedSpeaker)) {
            selectedSpeaker = '[S1]';
        }
    }

    // Event handlers for ChatInterface
    function handleMessagesUpdated(event: CustomEvent<Message[]>) {
        messages = event.detail;
            updateAvailableSpeakers();
        }

    function handleSpeakerChanged(event: CustomEvent<string>) {
        selectedSpeaker = event.detail;
    }

    // Event handlers for SoundEffectsPanel
    function handleInputUpdated(event: CustomEvent<string>) {
        currentInput = event.detail;
    }

    function handleLoadExample(event: CustomEvent<Message[]>) {
        messages = event.detail;
        updateAvailableSpeakers();
    }

    function handleEmptyChat() {
        messages = [];
        updateAvailableSpeakers();
    }

    // Event handlers for AudioControls
    function handleAudioRecorded(event: CustomEvent<string>) {
        recordedAudioUrl = event.detail;
    }

    function handleAudioUploaded(event: CustomEvent<string>) {
        uploadedAudioUrl = event.detail;
    }

    function handleRecordingStateChanged(event: CustomEvent<boolean>) {
        isRecording = event.detail;
    }

    function handleErrorUpdated(event: CustomEvent<string | null>) {
        error = event.detail;
    }

    // Event handlers for GenerationButton
    function handleAudioGenerated(event: CustomEvent<string>) {
            // Clean up any existing audio URL
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
        }
        audioUrl = event.detail;
        error = null;
    }

    function handleGenerationError(event: CustomEvent<string>) {
        error = event.detail;
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
                audioUrl = null;
        }
    }

    function handleProcessingStateChanged(event: CustomEvent<boolean>) {
        isProcessing = event.detail;
    }
</script>

<main class="container">
    <header>
        <h1>AI Text-to-Speech Generator</h1>
        <p class="subtitle">Transform your text to speech with Dia-1.6B model</p>
    </header>

    <section class="converter-section">
        <div class="main-input-container">
            <div class="text-input-section">
                <ChatInterface 
                    bind:messages={messages}
                    bind:currentInput={currentInput}
                    bind:availableSpeakers={availableSpeakers}
                    bind:selectedSpeaker={selectedSpeaker}
                    bind:editingMessageIndex={editingMessageIndex}
                    bind:editText={editText}
                    on:messagesUpdated={handleMessagesUpdated}
                    on:speakerChanged={handleSpeakerChanged}
                />
                <SoundEffectsPanel 
                    bind:selectedEffect={selectedEffect}
                    bind:currentInput={currentInput}
                    on:inputUpdated={handleInputUpdated}
                    on:loadExample={handleLoadExample}
                    on:emptyChat={handleEmptyChat}
                />
            </div>

            <div class="side-controls">
                <GenerationSettings 
                    bind:maxNewTokens={maxNewTokens}
                    bind:cfgScale={cfgScale}
                    bind:temperature={temperature}
                    bind:topP={topP}
                    bind:cfgFilterTopK={cfgFilterTopK}
                    bind:speedFactor={speedFactor}
                />
            </div>
        </div>

        <AudioControls 
            bind:isRecording={isRecording}
            bind:recordedAudioUrl={recordedAudioUrl}
            bind:uploadedAudioUrl={uploadedAudioUrl}
            bind:error={error}
            on:audioRecorded={handleAudioRecorded}
            on:audioUploaded={handleAudioUploaded}
            on:recordingStateChanged={handleRecordingStateChanged}
            on:errorUpdated={handleErrorUpdated}
        />

        <GenerationButton 
            bind:messages={messages}
            bind:isProcessing={isProcessing}
            bind:generationProgress={generationProgress}
            bind:recordedAudioUrl={recordedAudioUrl}
            bind:uploadedAudioUrl={uploadedAudioUrl}
            bind:maxNewTokens={maxNewTokens}
            bind:cfgScale={cfgScale}
            bind:temperature={temperature}
            bind:topP={topP}
            bind:cfgFilterTopK={cfgFilterTopK}
            bind:speedFactor={speedFactor}
            on:audioGenerated={handleAudioGenerated}
            on:generationError={handleGenerationError}
            on:processingStateChanged={handleProcessingStateChanged}
        />

        <AudioOutput 
            bind:audioUrl={audioUrl}
            bind:error={error}
        />
    </section>
</main>

<style>
    :global(body) {
        background-color: white;
        color: #333;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2.5rem;
    }

    header {
        text-align: center;
        margin-bottom: 3.5rem;
    }

    h1 {
        font-size: 2.2rem;
        color: #333;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666;
    }

    .converter-section {
        background: white;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        padding: 2rem;
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
    }

    .main-input-container {
        display: grid;
        grid-template-columns: minmax(300px, 1.2fr) minmax(250px, 1fr);
        gap: 2rem;
        margin-bottom: 1rem;
        align-items: stretch;
        min-height: 500px;
    }

    .text-input-section {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .side-controls {
        display: flex;
        flex-direction: column;
    }

    @media (max-width: 1200px) {
        .main-input-container {
            grid-template-columns: minmax(300px, 1.5fr) minmax(250px, 1fr);
            gap: 2rem;
        }
    }

    @media (max-width: 900px) {
        .container {
            padding: 1.5rem;
        }
        
        .converter-section {
            padding: 1.5rem;
        }

        .main-input-container {
            grid-template-columns: 1fr;
            gap: 1.5rem;
            min-height: auto;
        }

        .side-controls {
            order: -1;
        }
    }

    @media (max-width: 600px) {
        .container {
            padding: 1rem;
        }
        
        .converter-section {
            padding: 1rem;
        }

        h1 {
            font-size: 1.8rem;
        }

        .subtitle {
            font-size: 1rem;
        }
    }
</style> 