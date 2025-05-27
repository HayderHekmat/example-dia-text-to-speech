<script lang="ts">
    export let audioUrl: string | null = null;
    export let error: string | null = null;

    function downloadAudio() {
        if (!audioUrl) return;
        const link = document.createElement('a');
        link.href = audioUrl;
        link.download = `generated_audio_${new Date().getTime()}.wav`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
</script>

<div class="audio-input-section">
    <h3>Generated Audio</h3>
    <div class="audio-controls">
        <div class="audio-preview-container">
            {#if error}
                <p class="error-message">{error}</p>
            {:else if audioUrl}
                <div class="audio-preview">
                    <audio controls src={audioUrl} preload="auto">
                        Your browser does not support the audio element.
                    </audio>
                    <button 
                        class="download-button"
                        on:click={downloadAudio}
                        title="Download audio"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        Download
                    </button>
                </div>
            {:else}
                <p class="placeholder-text">Your generated audio will appear here</p>
            {/if}
        </div>
    </div>
</div>

<style>
    .audio-input-section {
        background: #fcfcfc;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }

    .audio-input-section h3 {
        font-size: 0.9rem;
        margin: 0 0 0.75rem 0;
        font-weight: 500;
    }

    .audio-controls {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        min-height: 45px;
        justify-content: flex-start;
    }

    .audio-preview-container {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 45px;
    }

    .audio-preview {
        width: 100%;
        max-width: 500px;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        align-items: center;
    }
    
    .audio-preview audio {
        width: 100%;
        height: 40px;
    }

    .download-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #333;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .download-button:hover {
        background: #222;
        transform: translateY(-1px);
    }

    .error-message {
        color: #e53e3e;
        font-weight: 500;
        margin: 0;
        font-size: 0.9rem;
    }

    .placeholder-text {
        color: #666;
        margin: 0;
        font-size: 0.9rem;
    }
</style> 