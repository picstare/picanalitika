FROM nginx:latest

# Remove the default configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy the NGINX configuration file
COPY nginx/nginx.conf /etc/nginx/conf.d/nginx.conf

# Copy the static files from the django
# COPY /backend/static/ /usr/share/nginx/html/static
COPY /backend/static/ /home/backend/static
# COPY /backend/staticfiles/ /home/app/staticfiles

RUN chown -R nginx:nginx /home/backend/static
RUN chmod -R 755 /home/backend/static

# RUN chown -R nginx:nginx /home/app/staticfiles
# RUN chmod -R 755 /home/app/staticfiles

# Set up logging directory
RUN touch /var/log/nginx/error.log
RUN chown -R nginx:nginx /var/log/nginx/

# Expose port 80
EXPOSE 80

# Start NGINX with error logging
CMD ["nginx", "-g", "error_log /var/log/nginx/error.log;", "-g", "daemon off;"]


# Set the permissions for the static folder
# RUN chown -R nginx:nginx /usr/share/nginx/html/static/
# RUN chmod -R 755 /usr/share/nginx/html/static/
