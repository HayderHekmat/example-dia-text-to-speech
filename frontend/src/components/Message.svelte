<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    
    export let message: { speaker: string; text: string };
    export let i: number;
    export let editingMessageIndex: number | null;
    export let editText: string;

    const dispatch = createEventDispatcher();

    function formatSpeakerName(speaker: string) {
        return speaker.replace('[S1]', 'Speaker 1').replace('[S2]', 'Speaker 2');
    }

    function handleKeyDown(event: KeyboardEvent) {
        if (event.key === 'Enter') dispatch('save');
        if (event.key === 'Escape') dispatch('cancel');
    }
</script>

<div class="message" data-speaker={message.speaker}>
    <div class="message-text">
        <div class="message-header">
            <div class="bubble-speaker">{formatSpeakerName(message.speaker)}</div>
            <div class="menu-container">
                <button class="menu-dots" aria-label="Message options">
                    <svg width="14" height="14" viewBox="0 0 16 16">
                        <circle cx="8" cy="3" r="1.5" />
                        <circle cx="8" cy="8" r="1.5" />
                        <circle cx="8" cy="13" r="1.5" />
                    </svg>
                </button>
                <div class="message-controls">
                    <button class="control-button" on:click={() => dispatch('edit', i)} aria-label="Edit message">edit</button>
                    <button class="control-button" on:click={() => dispatch('delete', i)} aria-label="Delete message">delete</button>
                </div>
            </div>
        </div>
        {#if editingMessageIndex === i}
            <input 
                bind:value={editText}
                class="edit-input"
                on:keydown={handleKeyDown}
            />
        {:else}
            <div class="bubble-content">{message.text}</div>
        {/if}
    </div>
</div>

<style>
    .message {
        display: flex;
        gap: 0.4rem;
        align-items: flex-start;
        animation: fadeIn 0.3s ease;
        padding: 0.1rem;
        flex-direction: row;
        position: relative;
    }

    .message[data-speaker="[S1]"] {
        flex-direction: row-reverse;
        justify-content: flex-start;
    }

    .message[data-speaker="[S2]"] {
        flex-direction: row;
        justify-content: flex-start;
    }

    .message-header {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        margin-bottom: 0.1rem;
        position: relative;
    }

    .menu-container {
        position: relative;
    }

    .menu-dots {
        background: none;
        border: none;
        padding: 0;
        cursor: pointer;
        opacity: 0.6;
        display: flex;
        align-items: center;
        fill: currentColor;
    }

    .menu-dots:hover {
        opacity: 1;
    }

    .menu-container:hover .message-controls {
        display: flex;
    }

    .message-controls {
        display: none;
        position: absolute;
        top: 100%;
        left: 0;
        background: white;
        border: 1px solid #eee;
        border-radius: 4px;
        padding: 0.3rem;
        flex-direction: column;
        gap: 0.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 10;
        min-width: 80px;
    }

    .message[data-speaker="[S1]"] .message-controls {
        left: auto;
        right: 0;
    }

    .control-button {
        background: none;
        border: none;
        color: #666;
        padding: 0;
        font-size: 0.7rem;
        cursor: pointer;
        opacity: 0.7;
    }

    .control-button:hover {
        opacity: 1;
        text-decoration: underline;
    }

    .bubble-speaker {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-bottom: 0.1rem;
        font-weight: 500;
    }

    .bubble-content {
        font-size: 0.9rem;
        line-height: 1.3;
    }

    .message[data-speaker="[S1]"] .bubble-speaker {
        color: #6366f1;
    }

    .message[data-speaker="[S2]"] .bubble-speaker {
        color: #ec4899;
    }

    .message-text {
        margin: 0;
        padding: 0.25rem 0.5rem;
        font-size: 0.9rem;
        line-height: 1.3;
        max-width: 70%;
        position: relative;
    }

    .message[data-speaker="[S1]"] .message-text {
        background: #e8e8fd;
        border-radius: 15px 15px 3px 15px;
    }

    .message[data-speaker="[S2]"] .message-text {
        background: #f5f5f5;
        border-radius: 15px 15px 15px 3px;
    }

    .message[data-speaker="[S1]"] .message-text::after {
        content: '';
        position: absolute;
        right: -8px;
        bottom: 0;
        width: 10px;
        height: 10px;
        background: #e8e8fd;
        clip-path: polygon(0 0, 0% 100%, 100% 100%);
    }

    .message[data-speaker="[S2]"] .message-text::after {
        content: '';
        position: absolute;
        left: -8px;
        bottom: 0;
        width: 10px;
        height: 10px;
        background: #f5f5f5;
        clip-path: polygon(100% 0, 100% 100%, 0 100%);
    }

    .edit-input {
        width: 100%;
        padding: 0.25rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 0.9rem;
        font-family: inherit;
    }

    .edit-input:focus {
        outline: none;
        border-color: #333;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style> 