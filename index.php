<?php
        $text=$_POST['in_data'];
        //$mode=$_POST['mode'];
        //$text='ewdew ewewf efefwe';
        // $last_line = exec('python lyrics_prediction.py "'.$mode.'" "'.$text.'"', $retval);
        
        // $text = preg_replace("/'/", "", $text);
        if ($text != ""){
                file_put_contents("temp.txt",$text);
                // echo $text
                // $mode="songdata.csv";
                $cmd="python lyrics_prediction.py ";//.$mode;
                // echo $cmd;
                $last_line = exec($cmd, $retval);
                echo $last_line;
        }
        
        // echo "hi";
?>