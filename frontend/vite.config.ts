import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	preview: {
		host: true,
		port: 4173,
		strictPort: true,
		allowedHosts: ['.koyeb.app']
	}
});
