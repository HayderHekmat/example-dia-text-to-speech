import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
    try {
        // Get the request body
        const body = await request.json();
        
        // Get the API URL from environment or use default
        const API_URL = process.env.API_URL || 'http://127.0.0.1:8000';
        
        // Forward the request to the backend
        const response = await fetch(`${API_URL}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            return json(
                { error: errorData?.detail || 'Failed to generate audio' },
                { status: response.status }
            );
        }

        // Get the audio blob from the response
        const blob = await response.blob();
        
        // Return the audio blob with appropriate headers
        return new Response(blob, {
            headers: {
                'Content-Type': 'audio/wav',
                'Content-Length': blob.size.toString()
            }
        });

    } catch (error) {
        console.error('Error in generate endpoint:', error);
        return json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}; 