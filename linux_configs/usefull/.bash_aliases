shopt -s expand_aliases
alias screen='if [ $MYVIMRC ] ; then echo '1' > ~/is_there_vim; else echo '0' > ~/is_there_vim; fi && /usr/bin/screen'

