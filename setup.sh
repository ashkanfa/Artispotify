mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"ashkan.farahani@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[deprecation]\n\
showPyplotGlobalUse = False\n\
" > ~/.streamlit/credentials.toml


echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
