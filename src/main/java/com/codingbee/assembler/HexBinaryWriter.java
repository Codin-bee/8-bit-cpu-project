package com.codingbee.assembler;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class HexBinaryWriter {
    public static void writeHexAsBinary(String hexString, String filename) {
        try (FileOutputStream fos = new FileOutputStream(filename)) {
            byte[] byteData = hexStringToByteArray(hexString);
            fos.write(byteData);
            System.out.println("Binary hex file written successfully: " + filename);
        } catch (IOException e) {
            System.err.println("Error writing binary hex file: " + e.getMessage());
        }
    }

    public static byte[] hexStringToByteArray(String hexString) {
        hexString = hexString.replaceAll("\\s", ""); // Remove spaces and newlines
        int len = hexString.length();
        byte[] data = new byte[len / 2];

        for (int i = 0; i < len; i += 2) {
            data[i / 2] = (byte) ((Character.digit(hexString.charAt(i), 16) << 4)
                    + Character.digit(hexString.charAt(i + 1), 16));
        }
        return data;
    }

    public static String loadFileAsString(String filename) {
        try {
            return new String(Files.readAllBytes(Paths.get(filename))).replace("\t", "").replace("\r\n", "\n"); // Normalize newlines
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            return "";
        }
    }

    public static void main(String[] args) {
//        args = new String[2];
//        args[0] = "src/main/resources/text.txt";
//        args[1] = "src/main/resources/output.hex";
        String original = loadFileAsString(args[0]);
        String hexString = Assembler.translate(original);
        String[] r = hexString.split("\n");
        System.out.println(r.length);
        for (String s : r) {
            System.out.println(s);
        }
        writeHexAsBinary(hexString, args[1]);
    }
}