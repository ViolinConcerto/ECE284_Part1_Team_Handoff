`timescale 1ns/1ps
module sfp_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset = 1, clear = 0, acc = 0, relu = 0;
    reg signed [15:0] data_in = 0;
    wire signed [15:0] data_out;
    sfp dut(.clk(clk), .reset(reset), .clear(clear), .acc(acc), .relu(relu),
            .in(data_in), .out(data_out));

    task tick;
        begin @(posedge clk); #1; @(negedge clk); end
    endtask
    task expect_value(input integer value);
        begin
            if (data_out !== value[15:0])
                $fatal(1, "SFP expected %0d, got %0d", value, data_out);
        end
    endtask
    initial begin
        @(negedge clk);
        tick(); reset = 0;
        data_in = -5; acc = 1; tick(); acc = 0;
        expect_value(-5);
        repeat (3) tick();
        expect_value(-5);
        data_in = 10; acc = 1; tick(); acc = 0;
        expect_value(5);
        relu = 1; tick(); relu = 0; expect_value(5);
        clear = 1; tick(); clear = 0; expect_value(0);
        data_in = -32768; acc = 1; tick(); acc = 0; expect_value(-32768);
        relu = 1; tick(); relu = 0; expect_value(0);
        data_in = 32767; acc = 1; tick(); acc = 0; expect_value(32767);
        reset = 1; tick(); reset = 0; expect_value(0);
        $display("PASS SFP");
        $finish;
    end
    initial begin #10000; $fatal(1, "SFP timeout"); end
endmodule
