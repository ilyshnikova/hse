#!/usr/bin/perl

use strict;
use warnings;

my $cluster_name = shift;
my $user_name = shift || 'olegts';

sub get_hosts {
	my $cluster_name = shift;

	open(my $fh, "</etc/clusters")
	or die "open failed: $!";

	my $host_names = [];
	while (my $line = <$fh>) {
		chomp($line);
		if ($line =~ /^$cluster_name (.*)$/) {
			$host_names = [split(/\s+/, $1)];
		}
	}

	close($fh);

	return $host_names;
}

my $hosts = get_hosts($cluster_name);

foreach my $host (@$hosts) {
	print "Sending to $host...\n";

	system("./install_to_host.sh $user_name\@$host:")
	and die "system failed";
}

