syntax on
filetype plugin indent on
imap <Nul> <C-X><C-U>
imap § <C-P>
imap <C-t> <Esc> :call PrepareTicket() <Enter>
vnoremap <silent><C-c> :call Comment()<Enter>
vnoremap <silent><C-u> :call UnComment()<Enter>
vnoremap <silent><C-t> :call PrepareTicket() <Enter>
map <C-x> : call SetConsole() <Enter>
imap <C-x> <ESC>: call SetConsole() <Enter> i
vnoremap <silent><C-x> :call SetConsole() <Enter>
map <C-n> :noh <Enter>
imap <C-n> <ESC>:noh <Enter> i
vnoremap <silent><C-n> :noh <Enter>
set pastetoggle=<C-x>
set mouse=a
set ttymouse=xterm2
set showtabline=2
set encoding=utf-8
set maxmempattern=1000000

map <C-f> :echo ''.expand('%').expand(':').expand(line('.')) <Enter>

map <ESC>[H <Home>
map <ESC>[F <End>
imap <ESC>[H <C-O><Home>
imap <ESC>[F <C-O><End>
cmap <ESC>[H <Home>
cmap <ESC>[F <End>

let $BASH_ENV = "~/.bash_aliases"

xnoremap p "_dP

autocmd BufEnter * :syntax sync fromstart
autocmd BufEnter *.py set ai sw=4 ts=4 sta et fo=croql softtabstop=4
autocmd BufEnter *.cpp set ai sw=4 ts=4 sta et fo=croql softtabstop=4
autocmd BufEnter *.h set ai sw=4 ts=4 sta et fo=croql softtabstop=4



com I call Import()

function Import()
        echo system('cat '.expand('%').expand(' | grep -E "^package" | sed "s/package/use/g"'))
endfunction

function SetConsole()
	if &number
		set nonumber
		set paste
	else
		echom "2"
		set number
		set nopaste
	endif
endfunction

function PrepareTicket()
	:s/^.*\/\([^\/]\+$\)/\1:\r\r/g
endfunction

function Comment()
        if &filetype != ''
                if &filetype == 'perl'
                        :s/^/#
                elseif &filetype == 'c'
                        :s/^/\/\/
                elseif &filetype == 'cpp'
                        :s/^/\/\/
                elseif &filetype == 'matlab'
                        :s/^/%
		else
		        :s/^/#
                endif
        else
                :s/^/#
        endif
endfunction

function UnComment()
        if &filetype != ''
                if &filetype == 'perl'
                        :s/^#/
                elseif &filetype == 'c'
                        :s/^\/\//
                elseif &filetype == 'cpp'
                        :s/^\/\//
                elseif &filetype == 'matlab'
                        :s/^%/
		else
		        :s/^#/
                endif
        else
                :s/^#/
        endif
endfunction



map <C-Up> mzMkzz`z
map <C-Down> mzMjzz`z
imap <C-Up> <ESC>mzMkzz`zi<Right>
imap <C-Down> <ESC>mzMjzz`zi<Right>

set showmatch
set number

:nnoremap <silent> <F5> :let _s=@/<Bar>:%s/\s\+$//e<Bar>:let @/=_s<Bar>:nohl<CR>
autocmd BufWritePre * :%s/\s\+$//e

set hlsearch
hi Search term=reverse ctermfg=7 ctermbg=1

set nocompatible
syntax enable
filetype plugin on
filetype indent on
let vimrplugin_screenplugin = 0
helptags ~/.vim/doc
let vimrplugin_underscore = 0
let vimrplugin_rnowebchunk = 0
let r_indent_align_args = 0

function CheckSyntax()
        if &filetype != ''
                let cmd=''
                if &filetype == 'perl'
                        let cmd='PERL5LIB=./ perl -wc '
                elseif &filetype == 'c'
                        let cmd='gcc -fsyntax-only -pedantic '
                elseif &filetype == 'cpp'
                        let cmd='g++ -fsyntax-only -pedantic '
                elseif &filetype == 'sh'
                        let cmd='bash -n '
                endif

                if cmd != ''
                        let output = system(cmd.expand('%').expand(' 2>&1 | grep -v INFO | grep -v WARN'))
                        echo strpart(output, 0, strlen(output)-1)
                else
                        echo 'Have no idea how to check sytax for filetype '.&ft
                endif
        else
                echo 'unknown file type or filetype plugin not loaded'
        endif
endfunction

function StyleCheck()
        if &filetype != ''
                let cmd=''
                if &filetype == 'perl'
                        let cmd='yabs-stylechecker '
                endif

                if cmd != ''
                        let output = system(cmd.expand('%').expand(' 2>&1 | grep -v INFO | grep -v WARN '))
                        echo strpart(output, 0, strlen(output)-1)
                else
                        echo 'Have no idea how to check style for filetype '.&ft
                endif
        else
                echo 'unknown file type or filetype plugin not loaded'
        endif
endfunction

function! DiffToggle()
    if &diff
        windo diffoff
    else
        windo diffthis
    endif
:endfunction

function! ExitSvndiff()
	let output = system('kill -9 $(ps aux | grep -E ''svn.*diff'' | awk ''{print $2;}'')')
:endfunction

function! ExpandDiff()
	set diffopt=filler,context:1000000,iwhite,horizontal
:endfunction

function! ShortDiff()
	set diffopt=filler,context:10,iwhite,horizontal
:endfunction

if &diff
	set diffopt=filler,context:1000000,iwhite,horizontal " filler is default and inserts empty lines for sync
	set t_Co=256
	highlight DiffAdd term=bold ctermbg=15
	highlight DiffDelete term=bold ctermbg=15
	highlight DiffChange term=bold ctermbg=14
	highlight DiffText term=bold ctermbg=11
endif
au VimEnter * if &diff | execute 'windo set wrap' | endif
au VimEnter * if &diff | execute 'windo set syntax=perl' | endif

map  <silent><C-E> :call ExpandDiff()<CR>
imap <silent><C-E> :call ExpandDiff()<CR>
vmap <silent><C-E> :call ExpandDiff()<CR>

if &diff
	map  <silent><C-W> :call ShortDiff()<CR>
	imap <silent><C-W> :call ShortDiff()<CR>
	vmap <silent><C-W> :call ShortDiff()<CR>

	map  <silent><C-D> :call DiffToggle()<CR>
	imap <silent><C-D> :call DiffToggle()<CR>
	vmap <silent><C-D> :call DiffToggle()<CR>
endif

map  <silent><C-Q> :call ExitSvndiff()<CR>
imap <silent><C-Q> :call ExitSvndiff()<CR>
vmap <silent><C-Q> :call ExitSvndiff()<CR>

map  <silent><C-S> :call StyleCheck()<CR>
imap <silent><C-S> :call StyleCheck()<CR>
vmap <silent><C-S> :call StyleCheck()<CR>

map  <silent><C-l> :syntax sync fromstart<CR>
imap <silent><C-l> :syntax sync fromstart<CR>
vmap <silent><C-l> :syntax sync fromstart<CR>



map  <silent><C-a> :call CheckSyntax()<CR>
imap <silent><C-a> :call CheckSyntax()<CR>
vmap <silent><C-a> :call CheckSyntax()<CR>

function OpenStored()
	let home_directory = system('echo $HOME')
	let home_directory = split(home_directory, '\n')[0]
	for line in readfile(home_directory.expand('/.vimstore'))
		for file_name in split(line, " ")
			execute 'tabnew' file_name
		endfor
	endfor
endfunction

com T call OpenStored()

command! -range V call setpos('.', [0,<line1>,0,0]) |
                    \ exe "normal V" |
                    \ call setpos('.', [0,<line2>,0,0])

set backspace=indent,eol,start
