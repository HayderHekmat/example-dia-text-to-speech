<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    
    const dispatch = createEventDispatcher();

    export let isRecording = false;
    export let recordedAudioUrl: string | null = null;
    export let uploadedAudioUrl: string | null = null;
    export let error: string | null = null;

    let mediaRecorder: MediaRecorder | null = null;
    let audioChunks: BlobPart[] = [];

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/mp3' });
                if (recordedAudioUrl) {
                    URL.revokeObjectURL(recordedAudioUrl);
                }
                recordedAudioUrl = URL.createObjectURL(audioBlob);
                dispatch('audioRecorded', recordedAudioUrl);
            };

            mediaRecorder.start();
            isRecording = true;
            dispatch('recordingStateChanged', isRecording);
        } catch (err) {
            console.error('Recording error:', err);
            error = 'Failed to start recording. Please check your microphone permissions.';
            dispatch('errorUpdated', error);
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            isRecording = false;
            dispatch('recordingStateChanged', isRecording);
        }
    }

    function handleFileUpload(event: Event) {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        
        if (file) {
            if (file.type !== 'audio/mp3' && file.type !== 'audio/mpeg') {
                error = 'Please upload an MP3 file.';
                dispatch('errorUpdated', error);
                return;
            }

            if (uploadedAudioUrl) {
                URL.revokeObjectURL(uploadedAudioUrl);
            }
            uploadedAudioUrl = URL.createObjectURL(file);
            error = null;
            dispatch('audioUploaded', uploadedAudioUrl);
            dispatch('errorUpdated', null);
        }
    }
</script>

<div class="audio-input-section">
    <h3>Reference Audio (optional)</h3>
    <div class="audio-controls">
        <div class="audio-button-group">
            <label class="audio-button" for="audio-file" title="Upload Audio" aria-label="Upload audio file">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <input
                    type="file"
                    id="audio-file"
                    accept="audio/mp3,audio/mpeg"
                    on:change={handleFileUpload}
                    class="hidden-input"
                />
            </label>

            {#if isRecording}
                <button 
                    class="audio-button recording"
                    on:click={stopRecording}
                    title="Stop Recording"
                    aria-label="Stop recording"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="6" y="6" width="12" height="12" rx="2"/>
                    </svg>
                </button>
            {:else}
                <button 
                    class="audio-button"
                    on:click={startRecording}
                    title="Start Recording"
                    aria-label="Start recording"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" y1="19" x2="12" y2="22"/>
                    </svg>
                </button>
            {/if}
        </div>

        <div class="audio-preview-container">
            {#if uploadedAudioUrl || recordedAudioUrl}
                <div class="audio-preview">
                    <audio controls src={uploadedAudioUrl || recordedAudioUrl} preload="auto">
                        Your browser does not support the audio element.
                    </audio>
                </div>
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

    .audio-button-group {
        display: flex;
        gap: 2.5rem;
        justify-content: center;
        margin-left: 2rem;
        margin-bottom: 0.5rem;
    }

    .audio-button {
        width: 130px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: white;
        border: 1px solid #eee;
        border-radius: 8px;
        color: #666;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .audio-button:hover {
        background: #f5f5f5;
        color: #333;
    }

    .audio-button.recording {
        background: #fee2e2;
        border-color: #fecaca;
        color: #dc2626;
    }

    .hidden-input {
        display: none;
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
        height: 36px;
    }
</style> 