FROM node:18-alpine AS builder

WORKDIR /app

# Since this is a rush monorepo app dependent on local packages,
# a full build is complex in Docker without 'rush deploy'.
# We will assume a 'rush deploy' output or simplified build for this demo.
# For now, we will just copy the source and try to build if possible,
# or use a pre-built approach.

# Given constraints, let's create a simplified Dockerfile that assumes
# 'rush deploy' was run OR we just build this specific app ignoring monorepo complexity
# (which might fail if it depends on local packages not published).

# STRATEGY: Copy EVERYTHING needed.
COPY . .

# Install rush
RUN npm install -g @microsoft/rush pnpm

# Install & Build
# Note: This might be slow and heavy.
RUN node common/scripts/install-run-rush.js install
RUN node common/scripts/install-run-rush.js build --to @flowgram.ai/demo-free-layout

# Runtime
FROM nginx:alpine
COPY --from=builder /app/apps/demo-free-layout/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
