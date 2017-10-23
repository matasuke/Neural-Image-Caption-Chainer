$(function(){
    
    $("#choose").change(function(){

        if(this.files.length){

            if($('#results').length){
                $('#results').empty();
            }

            var file = this.files[0];
            var canvas = document.getElementById('main');
            var ctx = canvas.getContext('2d');

            var image = new Image();

            var fr = new FileReader();

            fr.onload = function(evt){

                image.onload = function(){


                    //get the size of canvas;
                    var ch = canvas.height;
                    var cw = canvas.width;
                    
                    //get the size of image;
                    var image_height = image.naturalHeight;
                    var image_width = image.naturalWidth;

                    if(image_height >= image_width){
                        var mag = ch/image_height;
                    }else{
                        var mag = cw/image_width;
                    }

                    //resized image size;
                    var resized_height = image_height * mag;
                    var resized_width = image_width * mag;
                    ctx.fillStyle = "black";
                    ctx.fillRect(0, 0, cw, ch);
                    if(resized_height > resized_width){
                        var center = cw/2 - resized_width/2;
                        ctx.drawImage(image, center, 0, resized_width, resized_height);
                    }else{
                        var center = ch/2 - resized_height/2;
                        ctx.drawImage(image, 0, center, resized_width, resized_height);
                    }
                    

                    var targetFile = $('input[name=img]');
                    var fd = new FormData();
                    var target = targetFile.eq(0);
                    fd.append('file', $(target).prop("files")[0]);

                    $.ajax({
                        url: '/api',
                        type: 'POST',
                        contentType: 'image/jpeg',
                        dataType: 'json',
                        data: fd,
                        contentType: false,
                        processData: false,
                    })
                    .success(function(data, statusText, jqXHR){

                        var captions = data.results[0]
                        
                        $('#captions').empty();
                        
                        var head = '<tr class="active"><th class="col-md-1">Candidates</th><th class="col-md-4">Generated Captions</th></tr>';
                        
                        $('#captions').append(head);
                        
                        for(i = 0; i < captions.length; i++){
                            var caption = captions[i];
                            var element = '<tr><td>Candidate ' + i.toFixed() + '</td><td>' + caption + '</td></tr>';
                            $('#captions').append(element);

                        }
                    )
                    .fail(function(jqXHR, statusText, errorThrown){
                        console.log(errorThrown);
                        console.log(statusText);
                        console.log(jqXHR);
                    });
                }
            }
        fr.readAsDataURL(file);
        }
    });
});


drawImage = function(tag, img){
    
    var canvas = document.getElementById(tag);
    var ctx = canvas.getContext('2d');

    var image = new Image();

    image.onload = function(){

        //get the size of canvas;
        var ch = canvas.height;
        var cw = canvas.width;

        //get the size of image;
        var image_height = image.naturalHeight;
        var image_width = image.naturalWidth;

        if(image_height >= image_width){
            var mag = ch/image_height;
        }else{
            var mag = cw/image_width;
        }

        //resized image size;
        var resized_height = image_height * mag;
        var resized_width = image_width * mag;
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, cw, ch);
        if(resized_height > resized_width){
            var center = cw/2 - resized_width/2;
            ctx.drawImage(image, center, 0, resized_width, resized_height);
        }else{
            var center = ch/2 - resized_height/2;
            ctx.drawImage(image, 0, center, resized_width, resized_height);
        }
    }
    image.src = "data:image/jpg;base64," + img;
}
