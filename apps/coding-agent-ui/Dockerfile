FROM node:18-alpine AS builder

WORKDIR /app

COPY apps/coding-agent-ui apps/coding-agent-ui

WORKDIR /app/apps/coding-agent-ui
RUN npm install
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/apps/coding-agent-ui/dist /usr/share/nginx/html
COPY apps/coding-agent-ui/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
