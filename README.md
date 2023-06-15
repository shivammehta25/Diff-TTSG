# Diff-TTSG: Denoising probabilistic integrated speech and gesture synthesis


We present Diff-TTSG, the first diffusion model that jointly learns to synthesise speech and gestures together. Our method is probabilistic and non-autoregressive, and can be trained on small datasets from scratch. In addition, to showcase the efficacy of these systems and pave the way for their evaluation, we describe a set of careful uni- and multi-modal subjective tests for evaluating integrated speech and gesture synthesis systems.


<style type="text/css">
    .tg {
    border-collapse: collapse;
    border-color: #9ABAD9;
    border-spacing: 0;
  }

  .tg td {
    background-color: #EBF5FF;
    border-color: #9ABAD9;
    border-style: solid;
    border-width: 1px;
    color: #444;
    font-family: Arial, sans-serif;
    font-size: 14px;
    overflow: hidden;
    padding: 0px 20px;
    word-break: normal;
    font-weight: bold;
    vertical-align: middle;
    horizontal-align: center;
    white-space: nowrap;
  }

  .tg th {
    background-color: #000000;
    border-color: #9ABAD9;
    border-style: solid;
    border-width: 1px;
    color: #fff;
    font-family: Arial, sans-serif;
    font-size: 14px;
    font-weight: normal;
    overflow: hidden;
    padding: 0px 20px;
    word-break: normal;
    font-weight: bold;
    vertical-align: middle;
    horizontal-align: center;
    white-space: nowrap;
    padding: 10px;
    margin: auto;
  }

  .tg .tg-0pky {
    border-color: inherit;
    text-align: center;
    vertical-align: top,
  }

  .tg .tg-fymr {
    border-color: inherit;
    font-weight: bold;
    text-align: center;
    vertical-align: top
  }
  .slider {
  -webkit-appearance: none;
  width: 75%;
  height: 15px;
  border-radius: 5px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #409cff;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #409cff;
  cursor: pointer;
}

audio {
    width: 240px;
}

/* CSS */
.button-12 {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px 14px;
  font-family: -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
  border-radius: 6px;
  border: none;

  background: #6E6D70;
  box-shadow: 0px 0.5px 1px rgba(0, 0, 0, 0.1), inset 0px 0.5px 0.5px rgba(255, 255, 255, 0.5), 0px 0px 0px 0.5px rgba(0, 0, 0, 0.12);
  color: #DFDEDF;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
}

.button-12:focus {
  box-shadow: inset 0px 0.8px 0px -0.25px rgba(255, 255, 255, 0.2), 0px 0.5px 1px rgba(0, 0, 0, 0.1), 0px 0px 0px 3.5px rgba(58, 108, 217, 0.5);
  outline: 0;
}

video {
  margin: 1em;
}


</style>

<script src="transcripts.js"></script>

## Stimuli from the evaluation test

### Speech-only evaluation


> You walk around Dublin city centre and even if you try and strike up a conversation with somebody it's impossible because everyone has their headphones in. And again, I would listen to podcasts sometimes with my headphones in walking around the streets.
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">NAT</th>
      <th class="tg-0pky">Diff-TTSG</th>
      <th class="tg-0pky">T2-ISG</th>
      <th class="tg-0pky">Grad-TTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/GT_1_C4_2_eval_0137.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/TTSG_1_C4_2_eval_0137.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/ISG_1_C4_2_eval_0137.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/Grad-TTS_1_C4_2_eval_0137.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
  </tbody>
</table>

> And then a few weeks later after that my parents were away my granny was minding us and again I don't know why I told my brother to do this but I was like here.
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">NAT</th>
      <th class="tg-0pky">Diff-TTSG</th>
      <th class="tg-0pky">T2-ISG</th>
      <th class="tg-0pky">Grad-TTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/GT_2_C3_7_eval_0163.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/TTSG_2_C3_7_eval_0163.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/ISG_2_C3_7_eval_0163.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/Grad-TTS_2_C3_7_eval_0163.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
  </tbody>
</table>

> But I remember once my parents were just downstairs in the kitchen and this is when mobile phones just began coming out. So, like my oldest brother and my oldest sister had a mobile phone each I'm pretty sure.
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">NAT</th>
      <th class="tg-0pky">Diff-TTSG</th>
      <th class="tg-0pky">T2-ISG</th>
      <th class="tg-0pky">Grad-TTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/GT_3_C3_7_eval_0047.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/TTSG_3_C3_7_eval_0047.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/ISG_3_C3_7_eval_0047.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/Grad-TTS_3_C3_7_eval_0047.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
  </tbody>
</table>

> Eventually got to a point where I was like okay I need to stop doing this sort of stuff Like it just doesn't make any sense as to why because I was getting hurt like there was times where like, I was like tearing muscles and I never broke a bone which I'm pretty proud of.
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">NAT</th>
      <th class="tg-0pky">Diff-TTSG</th>
      <th class="tg-0pky">T2-ISG</th>
      <th class="tg-0pky">Grad-TTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/GT_4_C3_7_eval_0301.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/TTSG_4_C3_7_eval_0301.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/ISG_4_C3_7_eval_0301.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/Grad-TTS_4_C3_7_eval_0301.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
  </tbody>
</table>
<!-- 
> I would like replenish stock I would bring up stock for the off-license that sort of stuff So I was doing all the kind of the menial kind of jobs like the kind of boring tedious work that someone had to do.
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">NAT</th>
      <th class="tg-0pky">Diff-TTSG</th>
      <th class="tg-0pky">T2-ISG</th>
      <th class="tg-0pky">Grad-TTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/GT_5_C4_1_eval_0251.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/TTSG_5_C4_1_eval_0251.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/ISG_5_C4_1_eval_0251.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./stimuli/audio-only/Grad-TTS_5_C4_1_eval_0251.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
  </tbody>
</table> -->


### Gesture-only evaluation (no audio)

<video id="gesture-only-video" class="video-js" controls width="640" height="360">
    <source id="gesture-only-video-source" src="./stimuli/gesture-only/TTSG_1_C3_7_eval_0447.mp4" type='video/mp4' />
</video>

Currently loaded: <span id="playing-gesture-only" style="font-weight: bold;" > Diff-TTSG 1</span>

<blockquote>
  <p id="gesture-only-transcription">
    If you like touched it, it was excruciatingly sore. And I went up to the teachers I was like look I'm after like really damaging my finger I might have to go to the doctors.
  </p>
</blockquote>

<script>
gesture_only_video = document.getElementById('gesture-only-video')
gesture_only_video_source = document.getElementById('gesture-only-video-source')
gesture_only_span_text =  document.getElementById('playing-gesture-only')
gesture_only_transcript = document.getElementById('gesture-only-transcription')


transcript_gesture_only = {
  '1' : "If you like touched it, it was excruciatingly sore. And I went up to the teachers I was like look I'm after like really damaging my finger I might have to go to the doctors.",
  '2' : "When I was in primary school I used to have this ruler and I used to put it between desks and I used to push the tables together so the ruler would be between the two tables.",
  '3' : "Because you can actually you actually do feel the kind of the mental strains of social media and you know people depicting these perfect lives online and you're like oho.",
  '4' : "I mean it it's not that I'm against it it's just that I just don't have the time and I just sometimes I'm not bothered and that sort of stuff."
}


function play_video(filename, text){
    id = text[text.length - 1];

    gesture_only_video.pause();
    gesture_only_video_source.src = filename;
    gesture_only_span_text.innerHTML = text;
    gesture_only_transcript.innerHTML = transcript_gesture_only[id];
    gesture_only_video.load();
    gesture_only_video.play();

}
</script>

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Text prompt #</th>
    <th class="tg-0pky">NAT</th>
    <th class="tg-0pky">Diff-TTSG</th>
    <th class="tg-0pky">T2-ISG</th>
    <th class="tg-0pky">[Grad-TTS]+M</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/GT_1_C3_7_eval_0447.mp4', 'NAT 1')" >NAT 1</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/TTSG_1_C3_7_eval_0447.mp4', 'Diff-TTSG 1')" >Diff-TTSG 1</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/ISG_1_C3_7_eval_0447.mp4', 'T2-ISG 1')" >T2-ISG 1</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/Grad_1_C3_7_eval_0447.mp4', 'Grad-TTS + M 1')" >Grad-TTS + M 1</button></td>
  </tr>
  <tr>
    <td>2</td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/GT_2_C3_5_eval_0043.mp4', 'NAT 2')" >NAT 2</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/TTSG_2_C3_5_eval_0043.mp4', 'Diff-TTSG 2')" >Diff-TTSG 2</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/ISG_2_C3_5_eval_0043.mp4', 'T2-ISG 2')" >T2-ISG 2</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/Grad_2_C3_5_eval_0043.mp4', 'Grad-TTS + M 2')" >Grad-TTS + M 2</button></td>
  </tr>
  <tr>
    <td>3</td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/GT_3_C4_2_eval_0039.mp4', 'NAT 3')" >NAT 3</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/TTSG_3_C4_2_eval_0039.mp4', 'Diff-TTSG 3')" >Diff-TTSG 3</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/ISG_3_C4_2_eval_0039.mp4', 'T2-ISG 3')" >T2-ISG 3</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/Grad_3_C4_2_eval_0039.mp4', 'Grad-TTS + M 3')" >Grad-TTS + M 3</button></td>
  </tr>
  <tr>
    <td>4</td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/GT_4_C4_3_eval_0092.mp4', 'NAT 4')" >NAT 4</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/TTSG_4_C4_3_eval_0092.mp4', 'Diff-TTSG 4')" >Diff-TTSG 4</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/ISG_4_C4_3_eval_0092.mp4', 'T2-ISG 4')" >T2-ISG 4</button></td>
    <td><button class="button-12" role="button" onclick="play_video('stimuli/gesture-only/Grad_4_C4_3_eval_0092.mp4', 'Grad-TTS + M 4')" >Grad-TTS + M 4</button></td>
  </tr>
</tbody>
</table>


### Speech-and-gesture evaluation



<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Matched</th>
    <th class="tg-0pky">Mismatched</th>
  </tr>
</thead>
<tbody>
  <tr>
      <td> 
          <video id="speech-and-gesture-video-matched" class="video-js" controls width="500" height="282">
              <source id="speech-and-gesture-video-matched-source" src="./stimuli/speech-and-gesture/TTSG_C4_3_eval_0150_matched.mp4" type='video/mp4' />
          </video>
      </td>
      <td>
        <video id="speech-and-gesture-video-mismatched" class="video-js" controls width="500" height="282">
              <source id="speech-and-gesture-video-mismatched-source" src="./stimuli/speech-and-gesture/TTSG_C4_3_eval_0150_mismatched.mp4" type='video/mp4' />
          </video>
      </td>
  </tr>
</tbody>
</table>
<h6> *Note: Matched versus mismatched stimuli were not labelled in the study and presented in random order. </h6>


Currently loaded: <span id="playing-speech-and-gesture-span" style="font-weight: bold;" > Diff-TTSG 1</span>

<blockquote>
  <p id="speech-and-gesture-transcription">
    Yeah and then obviously there, there's certain choirs that come down to the church. There's a woman called, I can't remember her name. But she has an incredible voice. Like an amazing voice.
  </p>
</blockquote>


<script>

  speech_and_gesture_video_matched = document.getElementById('speech-and-gesture-video-matched')
  speech_and_gesture_video_matched_source = document.getElementById('speech-and-gesture-video-matched-source')

  speech_ang_gesture_video_mismatched = document.getElementById('speech-and-gesture-video-mismatched')
  speech_and_gesture_video_mismatched_source = document.getElementById('speech-and-gesture-video-mismatched-source')

  speech_and_gesture_span_text =  document.getElementById('playing-speech-and-gesture-span')
  speech_and_gesture_transcript = document.getElementById('speech-and-gesture-transcription')



  transcript_speech_and_gesture = {
    '1' : "Yeah and then obviously there, there's certain choirs that come down to the church. There's a woman called, I can't remember her name. But she has an incredible voice. Like an amazing voice.",
    '2' : "When you think about it, that you do as a child, it's just absolutely ridiculous that makes no sense. But you can always justify it back then because it just seemed like the fun right thing to do.",
    '3' : "You walk around Dublin city centre and even if you try and strike up a conversation with somebody it's impossible because everyone has their headphones in. And again, I would listen to podcasts sometimes with my headphones in walking around the streets.",
    '4' : "Just so this whole social networking stuff just really really annoys me and cause it just warps people's minds and people are so Fixated on their phones and that sort of stuff that I just hate that so much."
  }

  function play_speech_and_gesture_eval(matched_filename, mismatched_filename, text){
      id = text[text.length - 1];

      speech_and_gesture_video_matched.pause();
      speech_ang_gesture_video_mismatched.pause();

      speech_and_gesture_video_matched_source.src = matched_filename;
      speech_and_gesture_video_mismatched_source.src = mismatched_filename;

      speech_and_gesture_span_text.innerHTML = text;
      speech_and_gesture_transcript.innerHTML = transcript_speech_and_gesture[id];

      speech_and_gesture_video_matched.load();
      speech_ang_gesture_video_mismatched.load();
  }
</script>




<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Text prompt #</th>
    <th class="tg-0pky">NAT</th>
    <th class="tg-0pky">Diff-TTSG</th>
    <th class="tg-0pky">T2-ISG</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/GT_C4_3_eval_0150_matched.mp4', './stimuli/speech-and-gesture/GT_C4_3_eval_0150_mismatched.mp4' ,'NAT 1')" >NAT 1</button></td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/TTSG_C4_3_eval_0150_matched.mp4', './stimuli/speech-and-gesture/TTSG_C4_3_eval_0150_mismatched.mp4' ,'Diff-TTSG 1')" >Diff-TTSG 1</button></td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/ISG_C4_3_eval_0150_matched.mp4', './stimuli/speech-and-gesture/ISG_C4_3_eval_0150_mismatched.mp4' ,'T2-ISG 1')" >T2-ISG 1</button></td>
  </tr>
  <tr>
    <td>2</td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/GT_C3_7_eval_1074_matched.mp4', './stimuli/speech-and-gesture/GT_C3_7_eval_1074_mismatched.mp4' ,'NAT 2')" >NAT 2</button></td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/TTSG_C3_7_eval_1074_matched.mp4', './stimuli/speech-and-gesture/TTSG_C3_7_eval_1074_mismatched.mp4' ,'Diff-TTSG 2')" >Diff-TTSG 2</button></td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/ISG_C3_7_eval_1074_matched.mp4', './stimuli/speech-and-gesture/ISG_C3_7_eval_1074_mismatched.mp4' ,'T2-ISG 2')" >T2-ISG 2</button></td>
  </tr>
  <tr>
    <td>3</td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/GT_C4_2_eval_0137_matched.mp4', './stimuli/speech-and-gesture/GT_C4_2_eval_0137_mismatched.mp4' ,'NAT 3')" >NAT 3</button></td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/TTSG_C4_2_eval_0137_matched.mp4', './stimuli/speech-and-gesture/TTSG_C4_2_eval_0137_mismatched.mp4' ,'Diff-TTSG 3')" >Diff-TTSG 3</button></td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/ISG_C4_2_eval_0137_matched.mp4', './stimuli/speech-and-gesture/ISG_C4_2_eval_0137_mismatched.mp4' ,'T2-ISG 3')" >T2-ISG 3</button></td>
  </tr>
  <tr>
    <td>4</td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/GT_C4_2_eval_0011_matched.mp4', './stimuli/speech-and-gesture/GT_C4_2_eval_0011_mismatched.mp4' ,'NAT 4')" >NAT 4</button></td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/TTSG_C4_2_eval_0011_matched.mp4', './stimuli/speech-and-gesture/TTSG_C4_2_eval_0011_mismatched.mp4' ,'Diff-TTSG 4')" >Diff-TTSG 4</button></td>
    <td><button class="button-12" role="button" onclick="play_speech_and_gesture_eval('./stimuli/speech-and-gesture/ISG_C4_2_eval_0011_matched.mp4', './stimuli/speech-and-gesture/ISG_C4_2_eval_0011_mismatched.mp4' ,'T2-ISG 4')" >T2-ISG 4</button></td>
  </tr>
</tbody>
</table>



## Additional examples from proposed system (Diff-TTSG)

### Beat gestures 



<video id="beat-gesture-synthesis-video" class="video-js" controls width="640" height="360">
  <source id="beat-gesture-synthesis-video-source" src="./stimuli/synthesised_examples/beat_1.mp4" type='video/mp4' />
</video>

Currently loaded: <span id="beat-gesture-synthesis-video-span" style="font-weight: bold;" > Example 1</span>

<blockquote>
  <p id="best-gesture-transcript">
    And the train stopped, The door opened. I got out first, then Jack Kane got out, Ronan got out, Louise got out.
  </p>
</blockquote>

<script>
  beat_gesture_synthesis_video = document.getElementById('beat-gesture-synthesis-video')
  beat_gesture_synthesis_video_source = document.getElementById('beat-gesture-synthesis-video-source')
  beat_gesture_synthesis_video_span =  document.getElementById('beat-gesture-synthesis-video-span')
  beat_gesture_transcript = document.getElementById('best-gesture-transcript')

  transcript_beat_videos = {
    '1' : "And the train stopped, The door opened. I got out first, then Jack Kane got out, Ronan got out, Louise got out.",
    '2' : "Drop the car off in Miami , and that was fine So we had a big plan, and so literally one morning one by one we would literally run down the stairs with our bags and then just we just ran.",
    '3' : "It was so so good, absolutely fantastic. The other day we went to watch the movie I never thought I would like it this much but I was so wrong, it was amazing.",
    '4' : "Jim asked if we should do a group hug, it was a strange request but then, he insisted again this time I had to shout no please not again, no, never, I dont want to hug. Please leave me alone and I went out.",
    '5' : "I started counting one, two, three and then I turned back I see everyone ran and hid somewhere the game of hide and seek was so fun I am going to play it again later, with my niece and nephew.",
    '6' : "So I went to the other side and then suddenly I saw someone running towards me I didn't really understand who it was but then I heard the noise and I was like no screw this I just started running and I ran as fast as I possibly could."
  }


  function play_beat_video(filename, text){
    id = text[text.length - 1];

    beat_gesture_synthesis_video.pause();
    beat_gesture_synthesis_video_source.src = filename;
    beat_gesture_synthesis_video_span.innerHTML = text;
    beat_gesture_transcript.innerHTML = transcript_beat_videos[id];
    beat_gesture_synthesis_video.load();
    beat_gesture_synthesis_video.play();
  }
</script>

<table class="tg">
<tbody>
  <tr>
    <td>
      <button class="button-12" role="button" onclick="play_beat_video('./stimuli/synthesised_examples/beat_1.mp4', 'Example 1')">Example 1</button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_beat_video('./stimuli/synthesised_examples/beat_2.mp4', 'Example 2')">Example 2</button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_beat_video('./stimuli/synthesised_examples/beat_3.mp4', 'Example 3')">Example 3</button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_beat_video('./stimuli/synthesised_examples/beat_4.mp4', 'Example 4')">Example 4</button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_beat_video('./stimuli/synthesised_examples/beat_5.mp4', 'Example 5')">Example 5</button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_beat_video('./stimuli/synthesised_examples/beat_6.mp4', 'Example 6')">Example 6</button>
    </td>
  </tr>
</tbody>
</table>

### Positive-negative emotional pairs


<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Positive emotion</th>
    <th class="tg-0pky">Negative emotion</th>
  </tr>
</thead>
<tbody>
  <tr>
      <td> 
          <video id="positive-emotional-pair" class="video-js" controls width="500" height="282">
              <source id="positive-emotional-pair-source" src="./stimuli/positive-negative/pos_1.mp4" type='video/mp4' />
          </video>
      </td>
      <td>
        <video id="negative-emotional-pair" class="video-js" controls width="500" height="282">
              <source id="negative-emotional-pair-source" src="./stimuli/positive-negative/neg_1.mp4" type='video/mp4' />
          </video>
      </td>
  </tr>
</tbody>
</table>
Currently loaded: <span id="emotional-pair-text" style="font-weight: bold;" > Example 1</span>

<blockquote>
  <p id="emotional-pair-text-transcription">
    <b> Positive: </b> I went to a comedy show last night, and it was absolutely hilarious. The jokes were fresh and clever, and I laughed so hard my sides hurt. <br><b>Negative: </b>I tried meditating to relieve stress, but it just made me feel more anxious. I couldn't stop thinking about all the things I needed to do, and it felt like a waste of time.
  </p>
</blockquote>


<script>

  positive_emotional_pair = document.getElementById('positive-emotional-pair')
  positive_emotional_pair_source = document.getElementById('positive-emotional-pair-source')

  negative_emotional_pair = document.getElementById('negative-emotional-pair')
  negative_emotional_pair_source = document.getElementById('negative-emotional-pair-source')

  emotional_pair_text =  document.getElementById('emotional-pair-text')
  emotional_pair_text_transcription = document.getElementById('emotional-pair-text-transcription')



  transcript_pos_negative = {
    '1' : "<b> Positive: </b> I went to a comedy show last night, and it was absolutely hilarious. The jokes were fresh and clever, and I laughed so hard my sides hurt. <br><b>Negative: </b>I went to a comedy show last night, and it was painfully unfunny. The jokes were outdated and offensive, and I couldn't wait for it to be over.",
    '2' : "<b> Positive: </b> I planned a surprise birthday party for my partner, and everything went perfectly. The cake was delicious, the guests arrived on time, and my partner was genuinely surprised. <br><b>Negative: </b>I tried to plan a surprise birthday party for my partner, but everything went wrong. The cake was ruined, the guests arrived late, and my partner figured it out beforehand.",
    '3' : "<b> Positive: </b> I went shopping for a new outfit for an upcoming event, and I found the perfect ensemble. Everything was reasonably priced and fit like a dream, and I left feeling so satisfied. <br><b>Negative: </b>I went shopping for a new outfit for an upcoming event, but I couldn't find anything I liked. Everything was either too expensive or just didn't fit right, and I left feeling so frustrated.",
  }

  function play_positive_negative_pair(positive_filename, negative_filename, text){
      id = text[text.length - 1];

      positive_emotional_pair.pause();
      negative_emotional_pair.pause();

      positive_emotional_pair_source.src = positive_filename;
      negative_emotional_pair_source.src = negative_filename;

      emotional_pair_text.innerHTML = text;
      emotional_pair_text_transcription.innerHTML = transcript_pos_negative[id];

      positive_emotional_pair.load();
      negative_emotional_pair.load();
  }
</script>


<table class="tg">
<tbody>
  <tr>
    <td>
      <button class="button-12" role="button" onclick="play_positive_negative_pair('./stimuli/positive-negative/pos_1.mp4', './stimuli/positive-negative/neg_1.mp4' ,'Example 1')" >
        Example 1
      </button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_positive_negative_pair('./stimuli/positive-negative/pos_2.mp4', './stimuli/positive-negative/neg_2.mp4' ,'Example 2')" >
        Example 2
      </button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_positive_negative_pair('./stimuli/positive-negative/pos_3.mp4', './stimuli/positive-negative/neg_3.mp4' ,'Example 3')" >
        Example 3
      </button>
    </td>
  </tr>
</tbody>
</table>




## Importance of the diffusion model
To illustrate the importance of using diffusion in modelling both speech and motion, these stimuli compare synthesis from condition D-TTSG to synthesis directly from the &mu; values predicted by the D-TTSG decoder and Conformer.



<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"> &mu; (before diffusion)</th>
    <th class="tg-0pky">Final output (after diffusion)</th>
  </tr>
</thead>
<tbody>
  <tr>
      <td> 
          <video id="before-diffusion-mu" class="video-js" controls width="500" height="282">
              <source id="before-diffusion-mu-source" src="./stimuli/mu-synthesis/1_mu.mp4" type='video/mp4' />
          </video>
      </td>
      <td>
        <video id="after-diffusion" class="video-js" controls width="500" height="282">
              <source id="after-diffusion-source" src="./stimuli/mu-synthesis/1.mp4" type='video/mp4' />
          </video>
      </td>
  </tr>
</tbody>
</table>
Currently loaded: <span id="mu-diffusion-text" style="font-weight: bold;" > Example 1</span>

<blockquote>
  <p id="mu-diffusion-transcription">
    <b> Positive: </b> I went to a comedy show last night, and it was absolutely hilarious. The jokes were fresh and clever, and I laughed so hard my sides hurt. <br><b>Negative: </b>I tried meditating to relieve stress, but it just made me feel more anxious. I couldn't stop thinking about all the things I needed to do, and it felt like a waste of time.
  </p>
</blockquote>


<script>

  before_diffusion_mu = document.getElementById('before-diffusion-mu')
  before_diffusion_mu_source = document.getElementById('before-diffusion-mu-source')

  after_diffusion = document.getElementById('after-diffusion')
  after_diffusion_source = document.getElementById('after-diffusion-source')

  mu_diffusion_text =  document.getElementById('mu-diffusion-text')
  mu_diffusion_transcription = document.getElementById('mu-diffusion-transcription')



  transcript_mu_diffusion = {
    '1' : "But and again so that doesn't help people like myself and my friend who actually want to strike up a conversation with a genuine person out in the open because we don't want to go online we don't feel like we have to do that.",
    '2' : "You walk around Dublin city centre and even if you try and strike up a conversation with somebody it's impossible because everyone has their headphones in. And again, I would listen to podcasts sometimes with my headphones in walking around the streets.",
    '3' : "I mean it it's not that I'm against it it's just that I just don't have the time and I just sometimes I'm not bothered and that sort of stuff."
  }

  function play_mu_diffusion(mu_filename, final_filename, text){
      id = text[text.length - 1];

      before_diffusion_mu.pause();
      after_diffusion.pause();

      before_diffusion_mu_source.src = mu_filename;
      after_diffusion_source.src = final_filename;

      mu_diffusion_text.innerHTML = text;
      mu_diffusion_transcription.innerHTML = transcript_mu_diffusion[id];

      before_diffusion_mu.load();
      after_diffusion.load();
  }
</script>


<table class="tg">
<tbody>
  <tr>
    <td>
      <button class="button-12" role="button" onclick="play_mu_diffusion('./stimuli/mu-synthesis/1_mu.mp4', './stimuli/mu-synthesis/1.mp4' ,'Example 1')" >
        Example 1
      </button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_mu_diffusion('./stimuli/mu-synthesis/2_mu.mp4', './stimuli/mu-synthesis/2.mp4' ,'Example 2')" >
        Example 2
      </button>
    </td>
    <td>
      <button class="button-12" role="button" onclick="play_mu_diffusion('./stimuli/mu-synthesis/3_mu.mp4', './stimuli/mu-synthesis/3.mp4' ,'Example 3')" >
        Example 3
      </button>
    </td>
  </tr>
</tbody>
</table>