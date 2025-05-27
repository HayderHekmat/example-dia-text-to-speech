<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    
    const dispatch = createEventDispatcher();

    export let selectedEffect = '';
    export let currentInput = '';

    const soundEffects = [
        'burps',
        'clears throat',
        'coughs',
        'exhales',
        'gasps',
        'groans',
        'humming',
        'laughs',
        'mumbles',
        'screams',
        'sighs',
        'sneezes',
    ];

    function insertSoundEffect() {
        if (selectedEffect) {
            const effect = `(${selectedEffect})`;
            currentInput = currentInput + (currentInput.endsWith(' ') ? '' : ' ') + effect + ' ';
            selectedEffect = '';
            dispatch('inputUpdated', currentInput);
        }
    }

    function loadExampleDialogue1() {
        const exampleMessages = [
            { speaker: '[S1]', text: 'Hey, how was your weekend?' },
            { speaker: '[S2]', text: 'Amazing! Went hiking in the mountains. The view was breathtaking!' },
            { speaker: '[S1]', text: 'That sounds incredible! I need to get out more.' },
            { speaker: '[S2]', text: 'You should join me next time! The trail I found is perfect for beginners (laughs)' }
        ];
        dispatch('loadExample', exampleMessages);
    }

    function loadExampleDialogue2() {
        const exampleMessages = [
            { speaker: '[S1]', text: 'I could really use a French coffee right now.' },
            { speaker: '[S2]', text: 'Oh! I found this charming French café around the corner. So authentic!' },
            { speaker: '[S1]', text: 'Really? Do they have fresh pastries?' },
            { speaker: '[S2]', text: 'Yes! Their chocolate croissants are amazing! And the owner is from Paris (humming)' }
        ];
        dispatch('loadExample', exampleMessages);
    }

    function emptyChat() {
        dispatch('emptyChat');
    }
</script>

<div class="effect-controls">
    <select
        id="effect-select"
        bind:value={selectedEffect}
        class="effect-select"
        on:change={insertSoundEffect}
    >
        <option value="">Add sound effect...</option>
        {#each soundEffects as effect}
            <option value={effect}>{effect}</option>
        {/each}
    </select>
    <div class="example-buttons">
        <button class="example-button" on:click={loadExampleDialogue1}>
            Example: Hiking Chat
        </button>
        <button class="example-button" on:click={loadExampleDialogue2}>
            Example: Coffee Chat
        </button>
        <button class="example-button" on:click={emptyChat}>
            Empty Chat
        </button>
    </div>
</div>

<style>
    .effect-controls {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 0;
    }

    .effect-select {
        width: 100%;
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

    .effect-select:hover {
        background-color: #efefef;
        color: #333;
    }

    .effect-select:focus {
        outline: none;
        border-color: #333;
        color: #333;
    }

    .example-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.25rem;
    }

    .example-button {
        flex: 1;
        padding: 0.4rem;
        background: #f5f5f5;
        border: none;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #666;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .example-button:hover {
        background: #efefef;
        color: #333;
    }
</style> 