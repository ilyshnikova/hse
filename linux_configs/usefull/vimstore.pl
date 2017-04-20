#!/usr/bin/perl

use strict;
use warnings;

my $HOME = $ENV{'HOME'};
my $TARGET_FILE = "$HOME/.vimstore";

my $line_numbers = {map {$_ => 1} @ARGV};

open(my $ofh, ">$TARGET_FILE")
or die "open failed: $!";

my $current_line_number = 1;
while (my $line = <STDIN>) {
	chomp($line);
	$line = get_file_path($line);
	if (!scalar(keys %$line_numbers) || $line_numbers->{$current_line_number}) {
		print $ofh "$line\n";
	}
	++$current_line_number;
}

close($ofh);

sub get_file_path {
	my $filename = shift;
	my $result = `readlink -f $filename`;
	chomp($result);

	return $result;
}

