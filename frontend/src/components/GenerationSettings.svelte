<script lang="ts">
    export let maxNewTokens = 1024;  
    export let cfgScale = 3.0;      
    export let temperature = 1.2;   
    export let topP = 0.9;          
    export let cfgFilterTopK = 32;   
    export let speedFactor = 0.9;
</script>

<div class="generation-settings">
    <div class="frame-header">
        <h3>Generation Settings</h3>
        <p class="helper-text">Adjust parameters to control the generated speech</p>
    </div>
    <div class="settings-panel">
        <div class="parameter-control">
            <label for="max-tokens">
                Max New Tokens ({maxNewTokens})
                <div class="tooltip-container">
                    <button class="tooltip-trigger" on:mouseenter={(e) => {
                        const tooltip = e.currentTarget.nextElementSibling as HTMLElement;
                        if (!tooltip) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
                        tooltip.style.left = (rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2)) + 'px';
                    }}>?</button>
                    <div class="tooltip">Controls the maximum length of generated audio. Higher values allow for longer audio.</div>
                </div>
            </label>
            <input 
                type="range" 
                id="max-tokens"
                bind:value={maxNewTokens}
                min="860"
                max="3072"
                step="1"
            />
        </div>

        <div class="parameter-control">
            <label for="cfg-scale">
                CFG Scale ({cfgScale})
                <div class="tooltip-container">
                    <button class="tooltip-trigger" on:mouseenter={(e) => {
                        const tooltip = e.currentTarget.nextElementSibling as HTMLElement;
                        if (!tooltip) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
                        tooltip.style.left = (rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2)) + 'px';
                    }}>?</button>
                    <div class="tooltip">Controls how closely to follow the prompt. Higher values produce more faithful but potentially less natural output.</div>
                </div>
            </label>
            <input 
                type="range" 
                id="cfg-scale"
                bind:value={cfgScale}
                min="1"
                max="5"
                step="0.1"
            />
        </div>

        <div class="parameter-control">
            <label for="temperature">
                Temperature ({temperature})
                <div class="tooltip-container">
                    <button class="tooltip-trigger" on:mouseenter={(e) => {
                        const tooltip = e.currentTarget.nextElementSibling as HTMLElement;
                        if (!tooltip) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
                        tooltip.style.left = (rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2)) + 'px';
                    }}>?</button>
                    <div class="tooltip">Controls randomness in generation. Higher values produce more varied but potentially less consistent output.</div>
                </div>
            </label>
            <input 
                type="range" 
                id="temperature"
                bind:value={temperature}
                min="1"
                max="1.5"
                step="0.1"
            />
        </div>

        <div class="parameter-control">
            <label for="top-p">
                Top P ({topP})
                <div class="tooltip-container">
                    <button class="tooltip-trigger" on:mouseenter={(e) => {
                        const tooltip = e.currentTarget.nextElementSibling as HTMLElement;
                        if (!tooltip) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
                        tooltip.style.left = (rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2)) + 'px';
                    }}>?</button>
                    <div class="tooltip">Controls diversity in sampling. Higher values allow for more diverse outputs.</div>
                </div>
            </label>
            <input 
                type="range" 
                id="top-p"
                bind:value={topP}
                min="0.8"
                max="1"
                step="0.01"
            />
        </div>

        <div class="parameter-control">
            <label for="cfg-filter-top-k">
                CFG Filter Top K ({cfgFilterTopK})
                <div class="tooltip-container">
                    <button class="tooltip-trigger" on:mouseenter={(e) => {
                        const tooltip = e.currentTarget.nextElementSibling as HTMLElement;
                        if (!tooltip) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
                        tooltip.style.left = (rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2)) + 'px';
                    }}>?</button>
                    <div class="tooltip">Controls the number of top tokens to consider during generation. Higher values allow for more variety.</div>
                </div>
            </label>
            <input 
                type="range" 
                id="cfg-filter-top-k"
                bind:value={cfgFilterTopK}
                min="15"
                max="50"
                step="1"
            />
        </div>

        <div class="parameter-control">
            <label for="speed-factor">
                Speed Factor ({speedFactor})
                <div class="tooltip-container">
                    <button class="tooltip-trigger" on:mouseenter={(e) => {
                        const tooltip = e.currentTarget.nextElementSibling as HTMLElement;
                        if (!tooltip) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
                        tooltip.style.left = (rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2)) + 'px';
                    }}>?</button>
                    <div class="tooltip">Controls the speed of generated speech. Higher values produce faster speech.</div>
                </div>
            </label>
            <input 
                type="range" 
                id="speed-factor"
                bind:value={speedFactor}
                min="0.8"
                max="1"
                step="0.01"
            />
        </div>
    </div>
</div>

<style>
    .generation-settings {
        background: #fcfcfc;
        border-radius: 16px;
        padding: 1rem;
        position: sticky;
        top: 2rem;
        height: auto;
        min-height: calc(200px + 180px + 3rem);
        display: flex;
        flex-direction: column;
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

    .settings-panel {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        flex: 1;
        padding-right: 0.25rem;
        overflow: visible;
        position: relative;
        gap: 1.5rem;
    }

    .parameter-control {
        margin-bottom: 0;
        flex-shrink: 0;
        padding: 0.25rem 0;
        position: relative;
    }

    .parameter-control label {
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .tooltip-container {
        position: relative;
        display: inline-block;
    }

    .tooltip-trigger {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #eee;
        color: #666;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        cursor: help;
        transition: all 0.2s ease;
        border: none;
        padding: 0;
    }

    .tooltip-trigger:hover {
        background: #e0e0e0;
        transform: scale(1.1);
    }

    .tooltip {
        display: none;
        position: fixed;
        background: #333;
        color: white;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        width: 220px;
        z-index: 1000;
        pointer-events: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }

    .tooltip-trigger:hover + .tooltip {
        display: block;
    }

    .parameter-control input[type="range"] {
        width: 100%;
        height: 2px;
        background: #eee;
        border-radius: 1px;
        outline: none;
        -webkit-appearance: none;
    }

    .parameter-control input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        background: #333;
        border-radius: 50%;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .parameter-control input[type="range"]::-webkit-slider-thumb:hover {
        transform: scale(1.2);
    }
</style> 