<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    
    const dispatch = createEventDispatcher();

    export let messages: Array<{speaker: string, text: string}> = [];
    export let isProcessing = false;
    export let generationProgress = 0;
    export let recordedAudioUrl: string | null = null;
    export let uploadedAudioUrl: string | null = null;
    export let maxNewTokens = 1024;
    export let cfgScale = 3.0;
    export let temperature = 1.2;
    export let topP = 0.9;
    export let cfgFilterTopK = 32;
    export let speedFactor = 0.9;

    let progressInterval: number;

    async function handleGenerate() {
        if (messages.length === 0) return;
        
        isProcessing = true;
        generationProgress = 0;
        dispatch('processingStateChanged', isProcessing);
        
        try {
            // Start progress simulation
            progressInterval = setInterval(() => {
                if (generationProgress < 90) {
                    generationProgress += Math.random() * 15;
                    if (generationProgress > 90) generationProgress = 90;
                }
            }, 300);

            // Combine all messages into a single text, maintaining speaker labels
            const combinedText = messages.map(msg => `${msg.speaker}: ${msg.text}`).join('\n');
            
            // Define the request data type
            interface RequestData {
                text_input: string;
                max_new_tokens: number;
                cfg_scale: number;
                temperature: number;
                top_p: number;
                cfg_filter_top_k: number;
                speed_factor: number;
                audio_prompt_input?: {
                    sample_rate: number;
                    audio_data: number[];
                };
            }

            // Prepare the request data
            const requestData: RequestData = {
                text_input: combinedText,
                max_new_tokens: maxNewTokens,
                cfg_scale: cfgScale,
                temperature: temperature,
                top_p: topP,
                cfg_filter_top_k: cfgFilterTopK,
                speed_factor: speedFactor
            };

            // If there's an audio file, convert it to the format the backend expects
            const referenceAudioUrl = uploadedAudioUrl || recordedAudioUrl;
            if (referenceAudioUrl) {
                try {
                    // Create AudioContext
                    const audioContext = new AudioContext();
                    
                    // First convert the audio to a blob
                    const response = await fetch(referenceAudioUrl);
                    if (!response.ok) {
                        throw new Error(`Failed to fetch audio file: ${response.status} ${response.statusText}`);
                    }
                    const blob = await response.blob();
                    
                    // Convert blob to array buffer
                    const arrayBuffer = await blob.arrayBuffer();
                    
                    // Decode the audio data
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    // Get the Float32Array from the first channel and convert to regular array
                    const channelData = audioBuffer.getChannelData(0);
                    
                    // Ensure the data is in the correct range (-1 to 1)
                    const normalizedData = Array.from(channelData).map(x => 
                        Math.max(-1, Math.min(1, x)) // Clamp values between -1 and 1
                    );

                    // Resize the audio data to match the expected tensor dimensions
                    const targetLength = 3072; // The expected size from the error message
                    let resizedData;
                    
                    if (normalizedData.length > targetLength) {
                        // If data is too long, take a sample from the middle
                        const start = Math.floor((normalizedData.length - targetLength) / 2);
                        resizedData = normalizedData.slice(start, start + targetLength);
                    } else if (normalizedData.length < targetLength) {
                        // If data is too short, pad with zeros
                        resizedData = new Array(targetLength).fill(0);
                        resizedData.splice(0, normalizedData.length, ...normalizedData);
                    } else {
                        resizedData = normalizedData;
                    }
                    
                    requestData.audio_prompt_input = {
                        sample_rate: audioBuffer.sampleRate,
                        audio_data: resizedData
                    };
                    
                    // Close the audio context
                    await audioContext.close();
                    
                } catch (audioError: unknown) {
                    console.error('Detailed audio processing error:', audioError);
                    throw new Error(`Failed to process audio file: ${audioError instanceof Error ? audioError.message : 'Unknown error'}. Please try a different file.`);
                }
            }

            const response = await fetch('https://gothic-sara-ann-challenge-8bad5bca.koyeb.app/api/generate', { //TODO: Update this URL for local dev or deployment - keep /api/generate at the end
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                console.error('Server response error:', {
                    status: response.status,
                    statusText: response.statusText,
                    errorData: errorData
                });
                throw new Error(errorData?.detail || 'Failed to generate audio');
            }

            // Get the audio blob from the response
            const blob = await response.blob();
            
            if (blob.size === 0) {
                throw new Error('Generated audio is empty');
            }

            // Set progress to 100% when generation is complete
            generationProgress = 100;
            
            // Create a new URL for the audio blob
            const audioUrl = URL.createObjectURL(blob);

            // Test the audio to make sure it loads correctly
            const audioTest = new Audio(audioUrl);
            audioTest.onerror = (e) => {
                console.error('Audio loading error:', e);
                throw new Error('Error loading the generated audio');
            };
            
            dispatch('audioGenerated', audioUrl);

        } catch (err: unknown) {
            console.error('Generation error:', err);
            const errorMessage = err instanceof Error ? err.message : 'Failed to generate audio. Please try again.';
            dispatch('generationError', errorMessage);
        } finally {
            clearInterval(progressInterval);
            setTimeout(() => {
                isProcessing = false;
                generationProgress = 0;
                dispatch('processingStateChanged', isProcessing);
            }, 500); // Keep 100% visible briefly
        }
    }
</script>

<div class="controls">
    <button
        class="generate-button"
        class:processing={isProcessing}
        on:click={handleGenerate}
        disabled={!messages.length || isProcessing}
    >
        <span class="button-text">
            {#if isProcessing}
                Generating... {Math.round(generationProgress)}%
            {:else}
                Generate Audio
            {/if}
        </span>
        {#if isProcessing}
            <div class="progress-bar" style="width: {generationProgress}%"></div>
        {/if}
    </button>
</div>

<style>
    .controls {
        padding: 0 0 2rem 0;
    }

    .generate-button {
        background: #333;
        color: white;
        padding: 1.25rem;
        border: none;
        border-radius: 12px;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        width: 100%;
        font-weight: 500;
        letter-spacing: 0.02em;
        position: relative;
        overflow: hidden;
    }

    .generate-button:hover:not(:disabled) {
        background: #222;
        transform: translateY(-1px);
    }

    .generate-button:disabled:not(.processing) {
        background: #ddd;
        cursor: not-allowed;
    }

    .generate-button.processing {
        cursor: wait;
    }

    .progress-bar {
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        background: #222;
        transition: width 0.3s ease;
        z-index: 1;
    }

    .button-text {
        position: relative;
        z-index: 2;
    }
</style> 