if [[ -z $(ps aux | grep -E 'bash restore-ssh-sock.sh' | grep -v grep) ]]; then
	bash restore-ssh-sock.sh &
fi
