# Use Node.js as base image
FROM node:20-slim

# Install pnpm
RUN npm install -g pnpm

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=4173
ENV HOST=0.0.0.0
ENV NODE_ENV=production

RUN pnpm build

EXPOSE 4173

# Command to run the application using pnpm preview
CMD ["pnpm", "preview"]