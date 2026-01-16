FROM node:18-alpine AS builder

WORKDIR /app

# Copy rush configuration
COPY rush.json .
COPY common/config/rush common/config/rush
COPY common/scripts common/scripts

# Install rush
RUN npm install -g @microsoft/rush

# Copy package sources
COPY apps/coding-agent apps/coding-agent
COPY common/config common/config

# Install dependencies (simplified for Docker context, ideally we use rush deploy)
# For this specific app, we can just install local deps since it is standalone node app
WORKDIR /app/apps/coding-agent
RUN npm install
RUN npm run build

FROM node:18-alpine

WORKDIR /app
COPY --from=builder /app/apps/coding-agent/dist ./dist
COPY --from=builder /app/apps/coding-agent/package.json .
COPY --from=builder /app/apps/coding-agent/node_modules ./node_modules

EXPOSE 3001

CMD ["node", "dist/server.js"]
