while [ 1 ]; do
	ssh_socket=`find   /tmp/ssh* -type s  2>/dev/null |head -1 | tr -d '\n'`
	real_socket=`find $HOME/.ssh/ssh_auth_sock -printf %l`
	echo "real_socket=$real_socket"
	echo "ssh_socket=$ssh_socket"
	if [ "$ssh_socket" != "$real_socket" ]; then
		echo "reset sockets"
		ln -sf "$ssh_socket" "$HOME/.ssh/ssh_auth_sock"
	fi
	sleep 1
done

