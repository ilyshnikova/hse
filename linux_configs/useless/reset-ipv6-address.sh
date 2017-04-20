nohup grep 'dhclient' /etc/network/interfaces | sort | sed -E 's/\$IFACE/eth0/g; s/^([[:space:]]+)?(down|up)[[:space:]]+//I' | xargs -I {} sh -c 'sudo {}'
