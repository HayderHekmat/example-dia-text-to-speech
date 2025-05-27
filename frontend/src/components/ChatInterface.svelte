<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    import Message from './Message.svelte';
    
    const dispatch = createEventDispatcher();

    export let messages: Array<{speaker: string, text: string}> = [];
    export let currentInput = '';
    export let availableSpeakers: string[] = ['[S1]'];
    export let selectedSpeaker = '[S1]';
    export let editingMessageIndex: number | null = null;
    export let editText = '';

    function formatSpeakerName(speaker: string) {
        return speaker.replace('[S1]', 'Speaker 1').replace('[S2]', 'Speaker 2');
    }

    function handleEdit(event: CustomEvent<number>) {
        editingMessageIndex = event.detail;
        editText = messages[event.detail].text;
    }

    function handleDelete(event: CustomEvent<number>) {
        messages = messages.filter((_, i) => i !== event.detail);
        dispatch('messagesUpdated', messages);
    }

    function handleSave() {
        if (editingMessageIndex !== null) {
            messages[editingMessageIndex].text = editText;
            editingMessageIndex = null;
            editText = '';
            dispatch('messagesUpdated', messages);
        }
    }

    function handleCancel() {
        editingMessageIndex = null;
        editText = '';
    }

    function addMessage() {
        if (currentInput.trim()) {
            messages = [...messages, { speaker: selectedSpeaker, text: currentInput.trim() }];
            currentInput = '';
            editingMessageIndex = null;
            selectedSpeaker = selectedSpeaker === '[S1]' ? '[S2]' : '[S1]';
            dispatch('messagesUpdated', messages);
            dispatch('speakerChanged', selectedSpeaker);
        }
    }

    function handleKeyPress(event: KeyboardEvent) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            addMessage();
        }
    }
</script>

<div class="text-input-frame">
    <div class="frame-header">
        <h3>Chat Interface</h3>
        <p class="helper-text">Type messages for different speakers to generate conversation</p>
    </div>
    <div class="messages">
        {#each messages as message, i}
            <Message 
                {message}
                {i}
                {editingMessageIndex}
                {editText}
                on:edit={handleEdit}
                on:delete={handleDelete}
                on:save={handleSave}
                on:cancel={handleCancel}
            />
        {/each}
    </div>
    <div class="input-area">
        <div class="input-container">
            <select 
                bind:value={selectedSpeaker}
                class="speaker-select"
            >
                {#each availableSpeakers as speaker}
                    <option value={speaker}>{formatSpeakerName(speaker)}</option>
                {/each}
            </select>
            <textarea
                bind:value={currentInput}
                placeholder="Type your message..."
                rows="1"
                on:keydown={handleKeyPress}
            ></textarea>
            <button 
                class="send-button"
                on:click={addMessage}
                disabled={!currentInput.trim()}
                aria-label="Send message"
            >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
            </button>
        </div>
    </div>
</div>

<style>
    .text-input-frame {
        background: #fcfcfc;
        border-radius: 16px;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: calc(200px + 180px + 3rem);
        border: 1px solid #e5e7eb;
    }

    .frame-header {
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
        margin-bottom: 0.35rem;
    }

    .frame-header h3 {
        font-size: 0.9rem;
        margin: 0;
    }

    .helper-text {
        font-size: 0.8rem;
        color: #666;
    }

    .messages {
        flex: 1;
        min-height: 200px;
        max-height: 200px;
        overflow-y: auto;
        padding: 0.35rem 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }

    .input-area {
        margin-top: auto;
        padding-top: 0.5rem;
        border-top: 1px solid #eee;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .input-container {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        background: white;
        border: 1px solid #eee;
        border-radius: 12px;
        padding: 0.5rem;
    }

    .speaker-select {
        width: 100px;
        padding: 0.4rem;
        border: none;
        background: #f5f5f5;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #666;
        cursor: pointer;
        transition: all 0.2s ease;
        -webkit-appearance: none;
        -moz-appearance: none;
        appearance: none;
        background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23666%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E");
        background-repeat: no-repeat;
        background-position: right 0.5rem center;
        background-size: 0.65em auto;
        padding-right: 1.5rem;
    }

    .speaker-select:hover {
        background-color: #efefef;
        color: #333;
    }

    textarea {
        flex: 1;
        padding: 0.5rem;
        border: none;
        border-radius: 8px;
        font-size: 0.95rem;
        font-family: inherit;
        resize: none;
        transition: all 0.2s ease;
        background: transparent;
        min-height: 20px;
        max-height: 120px;
        line-height: 1.4;
    }

    textarea::placeholder {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        color: #666;
        opacity: 0.8;
    }

    textarea:focus {
        outline: none;
        background: transparent;
    }

    .send-button {
        background: #333;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .send-button:hover:not(:disabled) {
        background: #222;
        transform: translateY(-1px);
    }

    .send-button:disabled {
        background: #ddd;
        cursor: not-allowed;
    }
</style> 