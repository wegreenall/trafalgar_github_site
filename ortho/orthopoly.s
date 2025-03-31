let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/phd/programming_projects/ortho
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd measure.py
edit orthopoly.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
3wincmd h
wincmd w
wincmd w
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 31 + 134) / 268)
exe 'vert 2resize ' . ((&columns * 78 + 134) / 268)
exe 'vert 3resize ' . ((&columns * 78 + 134) / 268)
exe 'vert 4resize ' . ((&columns * 78 + 134) / 268)
argglobal
enew
file NERD_tree_1
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
wincmd w
argglobal
balt ~/phd/programming_projects/mercergp/kernels.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 1 - ((0 * winheight(0) + 37) / 75)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 06|
wincmd w
argglobal
if bufexists("basis_functions.py") | buffer basis_functions.py | else | edit basis_functions.py | endif
if &buftype ==# 'terminal'
  silent file basis_functions.py
endif
balt orthopoly.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 105 - ((37 * winheight(0) + 37) / 75)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 105
normal! 0
wincmd w
argglobal
if bufexists("~/phd/programming_projects/favard_kernels/randomkernels.py") | buffer ~/phd/programming_projects/favard_kernels/randomkernels.py | else | edit ~/phd/programming_projects/favard_kernels/randomkernels.py | endif
if &buftype ==# 'terminal'
  silent file ~/phd/programming_projects/favard_kernels/randomkernels.py
endif
balt basis_functions.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 36 - ((35 * winheight(0) + 37) / 75)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 36
normal! 012|
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 31 + 134) / 268)
exe 'vert 2resize ' . ((&columns * 78 + 134) / 268)
exe 'vert 3resize ' . ((&columns * 78 + 134) / 268)
exe 'vert 4resize ' . ((&columns * 78 + 134) / 268)
tabnext 1
badd +288 measure.py
badd +105 basis_functions.py
badd +1 orthopoly.py
badd +85 polynomials.py
badd +347 ~/phd/programming_projects/mercergp/kernels.py
badd +38 ~/phd/programming_projects/favard_kernels/randomkernels.py
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOF
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
