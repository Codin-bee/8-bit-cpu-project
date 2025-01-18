package com.codingbee.assembler;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;

public class Assembler {
    private static final HashMap<String, String> instructionSet = new HashMap<>();
    private static final HashMap<String, String> variables = new HashMap<>();
    private static final HashMap<String, String> loops = new HashMap<>();
    private static boolean initialized = false;
    private static int startingAddress = Integer.parseInt("0000", 16);
    private static int arithmeticAddress = Integer.parseInt("3E80", 16);
    private static int instructionNo = startingAddress;

    public static String translate(String input){
        if (!initialized){
            init();
            initialized = true;
        }
        //Remove indentation
        String[] commands = input.split("\n");

        List<String> translated = new ArrayList<>();
        String trans;
        for (String command : commands) {
            if (command.startsWith(";")) continue;
            command = command.split(";")[0];
            trans = translateCommand(command.replace("\t", "").toLowerCase(Locale.ROOT));
            if (trans != null) {
                //if (trans.equals("01")) translated.add(trans);
                translated.add(trans + "00");

            }
        }
        StringBuilder o = new StringBuilder();
        for (int i = 0; i < translated.size(); i++) {
            if (translated.get(i) != null) {
                o.append(translated.get(i));
                if (i != translated.size()-1){
                    o.append("\n");
                }
            }
        }
        return o.toString();
    }

    private static String translateCommand(String cmd){
        //System.out.println(Integer.toHexString(instructionNo));
        String trans = null;
        String[] bytes = cmd.split(" ");
        //System.out.println("l: " + bytes.length);
        if (bytes[0].startsWith("@")){
            variables.put(bytes[0], bytes[1]);
        } else if (bytes[0].startsWith("#")) {
            //DEFINE LOOP
            loops.put(bytes[0], String.format("%04X", instructionNo));//May change idk padding
            //System.out.println(String.format("%04X", instructionNo));

        } else {
            trans = instructionSet.get(bytes[0]);

            if (bytes.length != 1) {
                trans += " ";
            if (bytes[1].startsWith("@")) {
                trans += variables.get(bytes[1]);
            } else if (bytes[1].startsWith("#")) {
                //Loop shit
                trans += loops.get(bytes[1]);
            } else {
                trans += bytes[1]; //Just put this after the loop for correct padding
                for (int i = 0; i < 4 - bytes[1].length(); i++) {
                    trans += 0;
                }
            }
        }
        }
        instructionNo++;
        return trans;
    }

    private static void init(){
        instructionSet.put("prnt", "01 0000");
        instructionSet.put("lda", "04");
        instructionSet.put("ldb", "05");
        instructionSet.put("ldia", "06");
        instructionSet.put("ldib", "07");
        instructionSet.put("sta", "08");
        instructionSet.put("stb", "09");
        instructionSet.put("nop", "0A 0000");
        instructionSet.put("adda", "10");
        instructionSet.put("addb", "11");
        instructionSet.put("addia", "12");
        instructionSet.put("addib", "13");
        instructionSet.put("suba", "14");
        instructionSet.put("subb", "15");
        instructionSet.put("subia", "16");
        instructionSet.put("subib", "17");
        instructionSet.put("anda", "20");
//        instructionSet.put("andb", "21");
//        instructionSet.put("anda", "22");
//        instructionSet.put("andb", "23");
        instructionSet.put("ora", "24");
        instructionSet.put("orb", "25");
        instructionSet.put("oria", "26");
        instructionSet.put("orib", "27");
        instructionSet.put("xora", "28");
        instructionSet.put("xorb", "29");
        instructionSet.put("xoria", "2A");
        instructionSet.put("xorib", "2B");
        instructionSet.put("nota", "2C");
        instructionSet.put("notb", "2E");
        instructionSet.put("shla", "2F");
        instructionSet.put("shlb", "30");
        instructionSet.put("shra", "31");
        instructionSet.put("shrb", "32");
        instructionSet.put("rola", "33");
        instructionSet.put("rolb", "34");
        instructionSet.put("rora", "35");
        instructionSet.put("jmp", "40");
        instructionSet.put("je", "44");
        instructionSet.put("jne", "48");

    }

    @SuppressWarnings("unused")
    public static int getStartingAddress() {
        return startingAddress;
    }
    @SuppressWarnings("unused")
    public static void setStartingAddress(int startingAddress) {
        Assembler.startingAddress = startingAddress;
    }
    @SuppressWarnings("unused")
    public static int getArithmeticAddress() {
        return arithmeticAddress;
    }
    @SuppressWarnings("unused")
    public static void setArithmeticAddress(int arithmeticAddress) {
        Assembler.arithmeticAddress = arithmeticAddress;
    }
}
