from django.shortcuts import render
from django.http import HttpResponse
from .models import Post
from django.core.files.storage import FileSystemStorage
# data
# posts = [
#     {
#         'author': 'Shrikant K.',
#         'title': 'Django UI Framework',
#         'content': 'Learn how to make UI using Django',
#         'date_posted': 'August 27, 2018'
#     },
#     {
#         'author': 'Girish M.',
#         'title': 'Data Science',
#         'content': 'Learn Data Science Fundamentals.',
#         'date_posted': 'August 28, 2018'
#     }
# ]

# Create your views here.


def home(request):
    context = {'posts': Post.objects.all().order_by('-date_posted'), 'title': 'Home'}
    return render(request, 'blog/home.html', context)


def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})

def prediction(request): 
    return render(request, 'blog/prediction.html', {'title':'Prediction'})


def upload(request): 
    return render(request, 'blog/upload.html', {'title':'Upload'})

def name(request): 
    return render(request, 'blog/name.html', {'title':'Name'})

def name2(request): 
    return render(request, 'blog/name2.html', {'title':'Name2'})


def trial(request):
    if request.method=='POST':
        income=request.POST.get('income')
        age=request.POST.get('age')
        income = int(income)
        age = int(age)
        dist=request.POST.get('dist')
        exp=request.POST.get('exp')

        add=income+age

        print(add)

        return render(request, 'blog/name.html', {"addition" : add} )
    return render(request, 'blog/prediction.html')

def read(request):
    
    if request.method =='POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()   
        fs.save(uploaded_file.name, uploaded_file) 
    

        #latest_file = max(list_of_files, key = os.path.getctime)
        #print(latest_file)
        #return render(request, 'blog/name2.html')
    return render(request, 'blog/name2.html')    
