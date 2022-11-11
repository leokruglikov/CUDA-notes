# Introduction to CUDA
## Personal notes

### Disclaimers 
- <strong> This is a pre-alpha version </strong>
- I did my best to put all the possible references. I need to, however, mention that <strong> everything </strong> was taken from books and online open sources. 
- I am not a computer science professional. I am not even a computer science student. Thus there may be some major or minor inaccuracies.


### Comments
I want to share with you my personal notes on this topic. 

<img src="pngs/extract_pdf.png" alt="extract_pdf" style="height: 500px; width: 500px;"/>

At the moment, 
there are lot's of things to be completed and added, as initially, they were written for my personal use. 
I then decided, that adding some images and references would be the cause to share them publicly. 
The LaTeX document isn't __clean__ enough, regarding the references and paragraph intending. 

The author will do its best to add new chapters and section, as well as modify the new lacking features, meantioned above.
![The pdf document](cuda_recap.pdf "notes on cuda") and the source LaTeX code - `cuda_recap.tex`.

### ToDo's 
- [ x ] Check spelling/grammar.
- [ ] Add parallel algorithm - parallel scan.
- [ ] Add thread filtering (`__all()` & `__any()`)
- [ ] Constant memory, texture memory & peer access
- [ ] Cuda gdb


### Structure
The document is written in LaTeX. The contents are divided into different modules, which are included into the 
`cuda_recap.tex`. For compilation, LaTeX, together with all the necessary packages must be installed. The compilation is 
done the usual LaTeX way, together with the flag for the minted package `pdflatex --shell-escpe cuda_recap.tex`.

